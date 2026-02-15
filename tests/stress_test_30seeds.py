# ARC (Automatic Recovery Controller) - Self-Healing Neural Networks
# Copyright (c) 2026 Aryan Kaushik. All rights reserved.
#
# This file is part of ARC.
#
# ARC is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# ARC is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for
# more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ARC. If not, see <https://www.gnu.org/licenses/>.

"""
ARC 30-Seed Stress Test
"""
import sys
sys.path.insert(0, r'c:\Users\ARYAN\Desktop\NN')

import torch
import torch.nn as nn
from arc.intervention import WeightRollback, RollbackConfig

print('STRESS TEST: 30-SEED RECOVERY')
print('='*50)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
    def forward(self, x):
        return self.fc(x)

recovered = 0
for seed in range(30):
    torch.manual_seed(seed)
    model = Model()
    opt = torch.optim.Adam(model.parameters())
    rb = WeightRollback(model, opt, RollbackConfig(checkpoint_frequency=10), verbose=False)
    
    success = False
    for step in range(100):
        x = torch.randn(4, 10)
        loss = model(x).mean()
        
        if step == 50:
            loss = loss * float('inf')
        
        try:
            loss.backward()
        except:
            pass
        
        action = rb.step(loss)
        if action.rolled_back:
            success = True
        elif not torch.isnan(loss) and not torch.isinf(loss):
            opt.step()
        opt.zero_grad()
    
    if success:
        recovered += 1
    status = "OK" if success else "X"
    print(f'  Seed {seed:2d}: {status}')

print()
print('='*50)
print(f'RECOVERY RATE: {recovered}/30 ({recovered*100//30}%)')
print('='*50)
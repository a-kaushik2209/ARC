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

import re

with open("grand_results.txt", "r") as f:
    text = f.read()

# Find report
match = re.search(r"ARC GRAND UNIFIED BENCHMARK REPORT.*", text, re.DOTALL)
if match:
    report = match.group(0)
    # Print until dashed line after last item?
    # Actually just print likely correct length
    lines = report.split("\n")
    for line in lines:
        if "warning" in line.lower(): continue
        print(line)
else:
    print("Report not found in text.")
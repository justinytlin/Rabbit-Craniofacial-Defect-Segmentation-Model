#!/bin/zsh
# Double-click to start the defect segmentation app and open it in the browser.
cd "$(dirname "$0")"
exec /usr/bin/env python3 webapp.py --open

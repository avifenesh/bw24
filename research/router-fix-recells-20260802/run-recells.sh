#!/bin/bash
bash /tmp/run-board-q35.sh >> /tmp/recells-q35.log 2>&1
bash /tmp/run-board-q27.sh >> /tmp/recells-q27.log 2>&1
echo CELLS-DONE > /tmp/recells-done

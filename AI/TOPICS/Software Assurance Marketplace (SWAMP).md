### Software Assurance Marketplace (SWAMP) 
```
Secure Coding Practices, Automated Assessment Tools and the SWAMP
https://secdev.ieee.org/wp-content/uploads/2018/09/Secdev-Tutorial-SWAMP-2018-09-Miller-Heymann-camera-ready.pdf

1. Signing up for the SWAMP or accessing it via them github,Google, or InCommon credentials.
2. Uploading software to the SWAMP.
3. Running a variety of software assurance tools.
4. Viewing and interpreting the results.
5. Fixing problems found and iterating over the above steps

The goals for this tutorial are to teach software developers and designers to:
• Visualize code and software design from a security perspective.
• Learn specific techniques for writing secure code.
• Learn how software assurance tools can to help improve the security of their code.
• Learn about specific tools resources available to them and get initial experience using these resources.
```

### SOFTWARE ASSURANCE CONFERENCE 2018 https://continuousassurance.org/2018/11/07/software-assurance-conference-2018/

```
SWAMP INSTRUCTIONAL VIDEOS https://continuousassurance.org/2018/11/13/swamp-instructional-videos/
```
### 測試網址 https://www.mir-swamp.org/

https://platform.swampinabox.org/siab-latest-release/extract-installer.bash
```
#!/bin/bash

# This file is subject to the terms and conditions defined in
# 'LICENSE.txt', which is part of this source code distribution.
#
# Copyright 2012-2018 Software Assurance Marketplace

#
# Extract the SWAMP-in-a-Box installer.
# Assumes that this script is in the same directory as the tarballs.
#

BINDIR="$(dirname "$0")"
VERSION="1.33.4"
INSTALLER_TARBALL="$BINDIR/swampinabox-${VERSION}-installer.tar.gz"
INSTALLER_DIR="$BINDIR/swampinabox-${VERSION}-installer"

function exit_with_error() {
    echo ""
    echo "Error encountered. Check above for details." 1>&2
    exit 1
}

if [ ! -r "$INSTALLER_TARBALL" ]; then
    echo "Error: No such file (or file is not readable): $INSTALLER_TARBALL" 1>&2
    exit 1
fi

echo "Extracting: $INSTALLER_TARBALL"
echo ""
tar -xzv --no-same-owner --no-same-permissions -C "$BINDIR" -f "$INSTALLER_TARBALL" || exit_with_error
echo ""
echo "The SWAMP-in-a-Box installer can be found in: $INSTALLER_DIR"
```
```
SWAMP-in-a-Box Administrator Manual
https://platform.swampinabox.org/siab-latest-release/administrator_manual.html
```

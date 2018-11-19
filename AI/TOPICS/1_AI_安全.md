# AI and Security: Lessons, Challenges & Future Directions

```
Dawn Song
UC Berkeley

https://qconsf.com/sf2017/system/files/presentation-slides/dawn-qcon-nov-20172.pdf
```
# Conference
```
IEEE Cybersecurity Development Conference
September 30-October 2, 2018 | Cambridge, MA
Sponsored by the IEEE Computer Society Technical Committee on Security and Privacy

https://secdev.ieee.org/2018/home
```
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
```
SWAMP INSTRUCTIONAL VIDEOS https://continuousassurance.org/2018/11/13/swamp-instructional-videos/
```


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
# Models and applications 

- PassGAN: A Deep Learning Approach for Password Guessing
```
Briland Hitaj∗, Paolo Gasti†, Giuseppe Ateniese∗ and Fernando Perez-Cruz‡

https://arxiv.org/pdf/1709.00440.pdf

State-of-the-art password guessing tools, such as HashCat and John the Ripper, 
enable users to check billions of passwords per second against password hashes. 

In addition to performing straightforward dictionary attacks, 
these tools can expand password dictionaries using password generation rules, such as 
concatenation of words (e.g., "password123456") and leet speak (e.g., "password" becomes "p4s5w0rd"). 
Although these rules work well in practice, expanding them to model further passwords is a laborious task 
that requires specialized expertise. 

To address this issue, in this paper we introduce PassGAN, 
a novel approach that replaces human-generated password rules with theory-grounded machine learning algorithms. 

Instead of relying on manual password analysis, 
PassGAN uses a Generative Adversarial Network (GAN) to autonomously learn the distribution of real passwords 
from actual password leaks, and to generate high-quality password guesses. 

Our experiments show that this approach is very promising. 
When we evaluated PassGAN on two large password datasets, 
we were able to surpass rule-based and state-of-the-art machine learning password guessing tools. 

However, in contrast with the other tools, PassGAN achieved this result 
without any a-priori knowledge on passwords or common password structures. 

Additionally, when we combined the output of PassGAN with the output of HashCat, 
we were able to match 51%-73% more passwords than with HashCat alone. 

This is remarkable, because it shows that PassGAN can autonomously extract a considerable number of password properties that current state-of-the art rules do not encode.
```
- SSGAN: Secure Steganography Based on Generative Adversarial Networks
```
Haichao Shia,b, Jing Dongc, Wei Wangc, Yinlong Qianc, Xiaoyu Zhanga
a
Institute of Information Engineering, Chinese Academy of Sciences
b
School of Cyber Security, University of Chinese Academy of Sciences
cCenter for Research on Intelligent Perception and Computing, National Laboratory of Pattern
Recognition, Institute of Automation, Chinese Academy of Sciences

```

# Automatic Patch Generation

- Automatic Patch Generation for Security Functional Vulnerabilities with GAN
```
Ya Xiao, Danfeng (Daphne) Yao
Department of Computer Science, Virginia Tech
{yax99, danfeng} @vt.edu
```

- Automatic Patch Generation

- Learning to Repair Software Vulnerabilities with Generative Adversarial Networks[2018]

```
Jacob Harer, Onur Ozdemir, Tomo Lazovich, Christopher P. Reale, Rebecca L. Russell, Louis Y. Kim, Peter Chin
https://arxiv.org/pdf/1805.07475.pdf

Motivated by the problem of automated repair of software vulnerabilities, 
we propose an adversarial learning approach that maps from one discrete source domain to another target domain 
without requiring paired labeled examples or source and target domains to be bijections. 

We demonstrate that the proposed adversarial learning approach is an effective technique for repairing software vulnerabilities, performing close to seq2seq approaches that require labeled pairs. 

The proposed Generative Adversarial Network approach is application-agnostic in that 
it can be applied to other problems similar to code repair, such as grammar correction or sentiment translation.
```
- Automatic Patch Generation for Buffer Overflow Attacks
```
Alexey Smirnov Tzi-cker Chiueh
Computer Science Department  Stony Brook University

```

- Sound Patch Generation for Vulnerabilities[2018]
```
Zhen Huang University of Toronto
David Lie  University of Toronto

https://arxiv.org/pdf/1711.11136.pdf
```

- Automated Vulnerability Detection in Source Code Using Deep Representation Learning[2018]
```
Rebecca L. Russell, Louis Kim, Lei H. Hamilton, Tomo Lazovich, Jacob A. Harer, Onur Ozdemir, Paul M. Ellingwood, Marc W. McConley
https://arxiv.org/abs/1807.04320
```

- Automated software vulnerability detection with machine learning[2018]
```
Jacob A. Harer, Louis Y. Kim, Rebecca L. Russell, Onur Ozdemir, Leonard R. Kosta, Akshay Rangamani, 
Lei H. Hamilton, Gabriel I. Centeno, Jonathan R. Key, Paul M. Ellingwood, Erik Antelman, Alan Mackay, 
Marc W. McConley, Jeffrey M. Opper, Peter Chin, Tomo Lazovich
https://arxiv.org/abs/1803.04497
```

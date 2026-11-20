====================================================
Intel Mac OS X on a Mac Pro for qemu-system-x86_64
====================================================

This ``MacPro-opencore`` branch is a personal fork of QEMU. It runs Intel
Mac OS X 10.5 Leopard to 10.10 Yosemite on ``qemu-system-x86_64`` as a Mac
Pro (MacPro1,1 to MacPro5,1): the ``q35`` machine with OVMF and OpenCore,
an Apple SMC, and emulated ATI Radeon graphics cards that Mac OS X drives
with its own drivers, so nothing has to be installed in the guest. It is an
experiment and is not meant for upstream QEMU.

QEMU as a whole is released under the GNU General Public License, version
2. For full licensing details, consult the LICENSE file.

Download
========

Ready-made builds for macOS (Apple silicon and Intel) and 64-bit Windows,
with a GUI for setting up and starting machines:

  https://github.com/cat7/MacPro-opencore/releases/tag/MacPro-opencore-2026-11-20

All releases: https://github.com/cat7/MacPro-opencore/releases

The GUI is developed in https://github.com/cat7/Qemu-GUI-MacPro-Opencore.
Mac OS X itself and the SMC key are not included; the readme in the
download says where to start.

The machine
===========

=====================  ======================================================
Models                 MacPro1,1 to MacPro5,1; the processor is QEMU's
                       Penryn (1,1 to 3,1), Nehalem (4,1) or Westmere (5,1)
                       model, named as a Xeon the model shipped with
Mac OS X               10.5 to 10.10 (10.8 and later on MacPro3,1 and newer)
Firmware               OVMF (UEFI) with OpenCore on a generated boot disk
Graphics               ATI Radeon HD 5770 (10.6 to 10.10), ATI Radeon X1900
                       XT (10.5 to 10.7) or standard VGA; each Radeon has
                       its own UEFI GOP option ROM, built from ``roms/``
Storage                ICH9 AHCI (SATA)
Input                  USB keyboard and tablet; host USB devices
Acceleration           TCG everywhere; HVF on an Intel Mac; WHPX on Windows
=====================  ======================================================

The HD 5770
-----------

The HD 5770 (``-device ati-radeon-hd5770``) is modelled at the register
level: display controller, command processor and the 3D engine, which runs
the guest driver's shader programs. Drawing goes to the host's GPU, through
Metal on macOS and through Vulkan on Windows, with a CPU path for the rest.
Accelerated OpenGL in Mac OS X, Quartz Extreme and Core Animation work
with Apple's own driver.

Documentation
=============

Documentation can be found hosted online at
`<https://www.qemu.org/documentation/>`_. The documentation for the
current development version that is available at
`<https://www.qemu.org/docs/master/>`_ is generated from the ``docs/``
folder in the source tree, and is built by `Sphinx
<https://www.sphinx-doc.org/en/master/>`_.


Building
========

QEMU is multi-platform software intended to be buildable on all modern
Linux platforms, OS-X, Win32 (via the Mingw64 toolchain) and a variety
of other UNIX targets. The simple steps to build QEMU are:


.. code-block:: shell

  mkdir build
  cd build
  ../configure
  make

Additional information can also be found online via the QEMU website:

* `<https://wiki.qemu.org/Hosts/Linux>`_
* `<https://wiki.qemu.org/Hosts/Mac>`_
* `<https://wiki.qemu.org/Hosts/W32>`_


Submitting patches
==================

The QEMU source code is maintained under the GIT version control system.

.. code-block:: shell

   git clone https://gitlab.com/qemu-project/qemu.git

When submitting patches, one common approach is to use 'git
format-patch' and/or 'git send-email' to format & send the mail to the
qemu-devel@nongnu.org mailing list. All patches submitted must contain
a 'Signed-off-by' line from the author. Patches should follow the
guidelines set out in the `style section
<https://www.qemu.org/docs/master/devel/style.html>`_ of
the Developers Guide.

Additional information on submitting patches can be found online via
the QEMU website:

* `<https://wiki.qemu.org/Contribute/SubmitAPatch>`_
* `<https://wiki.qemu.org/Contribute/TrivialPatches>`_

The QEMU website is also maintained under source control.

.. code-block:: shell

  git clone https://gitlab.com/qemu-project/qemu-web.git

* `<https://www.qemu.org/2017/02/04/the-new-qemu-website-is-up/>`_

A 'git-publish' utility was created to make above process less
cumbersome, and is highly recommended for making regular contributions,
or even just for sending consecutive patch series revisions. It also
requires a working 'git send-email' setup, and by default doesn't
automate everything, so you may want to go through the above steps
manually for once.

For installation instructions, please go to:

*  `<https://github.com/stefanha/git-publish>`_

The workflow with 'git-publish' is:

.. code-block:: shell

  $ git checkout master -b my-feature
  $ # work on new commits, add your 'Signed-off-by' lines to each
  $ git publish

Your patch series will be sent and tagged as my-feature-v1 if you need to refer
back to it in the future.

Sending v2:

.. code-block:: shell

  $ git checkout my-feature # same topic branch
  $ # making changes to the commits (using 'git rebase', for example)
  $ git publish

Your patch series will be sent with 'v2' tag in the subject and the git tip
will be tagged as my-feature-v2.

Bug reporting
=============

The QEMU project uses GitLab issues to track bugs. Bugs
found when running code built from QEMU git or upstream released sources
should be reported via:

* `<https://gitlab.com/qemu-project/qemu/-/issues>`_

If using QEMU via an operating system vendor pre-built binary package, it
is preferable to report bugs to the vendor's own bug tracker first. If
the bug is also known to affect latest upstream code, it can also be
reported via GitLab.

For additional information on bug reporting consult:

* `<https://wiki.qemu.org/Contribute/ReportABug>`_


ChangeLog
=========

For version history and release notes, please visit
`<https://wiki.qemu.org/ChangeLog/>`_ or look at the git history for
more detailed information.


Contact
=======

The QEMU community can be contacted in a number of ways, with the two
main methods being email and IRC:

* `<mailto:qemu-devel@nongnu.org>`_
* `<https://lists.nongnu.org/mailman/listinfo/qemu-devel>`_
* #qemu on irc.oftc.net

Information on additional methods of contacting the community can be
found online via the QEMU website:

* `<https://wiki.qemu.org/Contribute/StartHere>`_

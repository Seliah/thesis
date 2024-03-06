# config

This folder contains configuration files for different programs

The ".service" files are systemd "service unit" definitions. These definitions describe microservices/long running processes that are supervised by systemd. See <https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html>. With these definitions, the implemented programs can be deployed to run long term on any systemd-based system.

Also the HTOP config file (htoprc) is included to reproduce the metrics view.

### Custom operators for DALI

To debug the plugin with gdb

```
gdb python
break main
run test.py
## breakpoint 1 is hit
set stop-on-solib 1
cont
## stops when any new shared library is loaded
info shared
```
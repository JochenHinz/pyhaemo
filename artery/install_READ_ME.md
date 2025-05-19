
Pyvista could give you this error when the plot function is called: 

"""
libGL error: MESA-LOADER: failed to open iris: /usr/lib/dri/iris_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
"""

reulting in a seg fault.

How to fix it?

Write on your terminal this: find / -name libstdc++.so.6 2>/dev/null
This will search for the path of the library libstdc++.so.6.

Then add to your .bashrc the path of the library, for instance the following line: 
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

For more details please check:
https://stackoverflow.com/questions/71010343/cannot-load-swrast-and-iris-drivers-in-fedora-35/72200748#72200748
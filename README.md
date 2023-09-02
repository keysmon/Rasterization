# Rasterization
rendering system for triangulated 3D models based on rasterization



Load and Render a 3D model 
------------------
![](img/orthographic/simple.png)
----------------

Render the bunny using a wireframe; only the edges of the triangles are drawn.
![](img/orthographic/wireframe.png)


Shading
-------------

In **Flat Shading** each triangle is rendered using a unique normal (i.e. the normal of all the fragments that compose a triangle is simply the normal of the plane that contains it).

In **Per-Vertex Shading** the normals are specified on the vertices of the mesh, the color is computed for each vertex, and then interpolated in the interior of the triangle.

flat shading

![](img/orthographic/flat_shading.png)

vertex shading

![](img/orthographic/pv_shading.png)


Object Transformation 
----------------------------------

animation for flat shading

![](img/orthographic/bunny.gif)


Camera 
-------------------------------

flat shading and perspective camera

![](img/perspective/flat_shading.png)

vertex shading and perspective camera

![](img/perspective/pv_shading.png)

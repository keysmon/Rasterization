// C++ include
#include <iostream>
#include <string>
#include <vector>

// Utilities for the Assignment
#include "raster.h"

#include <gif.h>
#include <fstream>

#include <Eigen/Geometry>
// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"
using namespace std;
using namespace Eigen;

//Image height
const int H = 480;

//Camera settings
const double near_plane = 1.5; //AKA focal length
const double far_plane = near_plane * 100;
const double field_of_view = 0.7854; //45 degrees
const double aspect_ratio = 1.5;
const bool is_perspective = true;
const Vector3d camera_position(0, 0, 3);
const Vector3d camera_gaze(0, 0, -1);
const Vector3d camera_top(0, 1, 0);

//Object
const std::string data_dir = DATA_DIR;
const std::string mesh_filename(data_dir + "bunny.off");
//const std::string mesh_filename(data_dir + "bumpy_cube.off");

MatrixXd vertices; // n x 3 matrix (n points)
MatrixXi facets;   // m x 3 matrix (m triangles)

//Material for the object
const Vector3d obj_diffuse_color(0.5, 0.5, 0.5);
const Vector3d obj_specular_color(0.2, 0.2, 0.2);
const double obj_specular_exponent = 256.0;

//Lights
std::vector<Vector3d> light_positions;
std::vector<Vector3d> light_colors;
//Ambient light
const Vector3d ambient_light(0.3, 0.3, 0.3);

//Fills the different arrays
void setup_scene()
{
    //Loads file
    std::ifstream in(mesh_filename);
    if (!in.good())
    {
        std::cerr << "Invalid file " << mesh_filename << std::endl;
        exit(1);
    }
    std::string token;
    in >> token;
    int nv, nf, ne;
    in >> nv >> nf >> ne;
    vertices.resize(nv, 3);
    facets.resize(nf, 3);
    for (int i = 0; i < nv; ++i)
    {
        in >> vertices(i, 0) >> vertices(i, 1) >> vertices(i, 2);
    }
    for (int i = 0; i < nf; ++i)
    {
        int s;
        in >> s >> facets(i, 0) >> facets(i, 1) >> facets(i, 2);
        assert(s == 3);
    }

    //Lights
    light_positions.emplace_back(8, 8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(6, -8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(4, 8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(2, -8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(0, 8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(-2, -8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(-4, 8, 0);
    light_colors.emplace_back(16, 16, 16);
}

Matrix4f ortho(float left,float right,float bottom,float top,float z_near,float z_far){
    Matrix4f mat = Matrix4f::Identity();
    mat(0,0) = 2.f/(right - left);
    mat(1,1) = 2.f /(top - bottom);
    mat(2,2) = 2.f /(z_near-z_far);
    mat(0,3) = -(right+left)/(right - left);
    mat(1,3) = -(top+bottom)/(top-bottom);
    mat(2,3) = -(z_near+z_far)/(z_near-z_far);
    return mat;
}



void build_uniform(UniformAttributes &uniform)
{
    //TODO: setup uniform

    //TODO: setup camera, compute w, u, v
    Vector3d w = -camera_gaze.normalized();
    Vector3d u = camera_top.cross(w).normalized();
    Vector3d v = w.cross(u);

    //TODO: compute the camera transformation
    Matrix4f transformation;
    transformation <<
    u(0),v(0),w(0),camera_position(0),
    u(1),v(1),w(1),camera_position(1),
    u(2),v(2),w(2),camera_position(2),
    0,0,0,1;
    transformation << transformation.inverse();
    uniform.view = transformation;
    
    //TODO: setup projection matrix
    float ortho_t = near_plane * tan(field_of_view/2.f);
    float ortho_b = -ortho_t;
    float ortho_r = ortho_t * aspect_ratio;
    float ortho_l = -ortho_r;
    
    uniform.projective = ortho(ortho_l,ortho_r,ortho_b,ortho_t,-near_plane,-far_plane);
    
    Matrix4d P;
    if (is_perspective)
    {
        uniform.projective =  uniform.projective * transformation;
    }
    else
    {
    }

}



void simple_render(Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{
    
    
    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;

    program.VertexShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        //TODO: fill the shader
        VertexAttributes out;
        out.position = uniform.projective * va.position;
        
        return out;
    };

    program.FragmentShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        //TODO: fill the shader
        return FragmentAttributes(1, 0, 0);
    };

    program.BlendingShader = [](const FragmentAttributes &fa, const FrameBufferAttributes &previous) {
        //TODO: fill the shader
        return FrameBufferAttributes(fa.color[0] * 255, fa.color[1] * 255, fa.color[2] * 255, fa.color[3] * 255);
    };

    std::vector<VertexAttributes> vertex_attributes;
    //TODO: build the vertex attributes from vertices and facets
    
    //cout << "total" << facets.rows()<<endl;
    for (int i = 0; i < facets.rows(); ++i){
        int p1 = facets(i,0);
        int p2 = facets(i,1);
        int p3 = facets(i,2);
        
        VertexAttributes v1 = VertexAttributes(vertices(p1,0),vertices(p1,1),vertices(p1,2));
        VertexAttributes v2 = VertexAttributes(vertices(p2,0),vertices(p2,1),vertices(p2,2));
        VertexAttributes v3 = VertexAttributes(vertices(p3,0),vertices(p3,1),vertices(p3,2));
        vertex_attributes.push_back(v1);
        vertex_attributes.push_back(v2);
        vertex_attributes.push_back(v3);
    }
    rasterize_triangles(program, uniform, vertex_attributes, frameBuffer);
}

Matrix4f compute_rotation(const double alpha)
{
    //TODO: Compute the rotation matrix of angle alpha on the y axis around the object barycenter
    Matrix4f res;

    res << cos(alpha), 0, sin(alpha), 0,
        0, 1, 0, 0,
        -sin(alpha), 0, cos(alpha), 0,
        0, 0, 0, 1;
   
    return res;
}



void wireframe_render(const double alpha, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{
    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;
    
    Matrix4f trafo = compute_rotation(alpha);

    program.VertexShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        VertexAttributes out;
        out.position = -uniform.projective * va.trafo* va.position  ;
        return out;
    };

    program.FragmentShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        //TODO: fill the shader
        return FragmentAttributes(1, 0, 0);
    };

    program.BlendingShader = [](const FragmentAttributes &fa, const FrameBufferAttributes &previous) {
        //TODO: fill the shader
        return FrameBufferAttributes(fa.color[0] * 255, fa.color[1] * 255, fa.color[2] * 255, fa.color[3] * 255);
    };

    std::vector<VertexAttributes> vertex_attributes;

    //TODO: generate the vertex attributes for the edges and rasterize the lines
    //TODO: use the transformation matrix
    //TODO: build the vertex attributes from vertices and facets
    
    //cout << "total" << facets.rows()<<endl;
    for (int i = 0; i < facets.rows(); ++i){
        int p1 = facets(i,0);
        int p2 = facets(i,1);
        int p3 = facets(i,2);
        
        VertexAttributes v1 = VertexAttributes(vertices(p1,0),vertices(p1,1),vertices(p1,2));
        VertexAttributes v2 = VertexAttributes(vertices(p2,0),vertices(p2,1),vertices(p2,2));
        VertexAttributes v3 = VertexAttributes(vertices(p3,0),vertices(p3,1),vertices(p3,2));
        v1.trafo = trafo;
        v2.trafo = trafo;
        v3.trafo = trafo;
        vertex_attributes.push_back(v2);
        vertex_attributes.push_back(v1);

        vertex_attributes.push_back(v3);
        vertex_attributes.push_back(v2);

        vertex_attributes.push_back(v1);
        vertex_attributes.push_back(v3);


    }
    rasterize_lines(program, uniform, vertex_attributes, 0.5, frameBuffer);
}

void get_shading_program(Program &program)
{
    
    program.VertexShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        //TODO: transform the position and the normal
        //TODO: compute the correct lighting
        
        VertexAttributes out;
        out.position = - uniform.projective * va.trafo *va.position;
        Vector3d p,N;
        p << va.position(0),va.position(1),va.position(2);
        N = va.normal;
        Vector3d lights_color(0, 0, 0);


        // Punctual lights contribution (direct lighting)
        for (int i = 0; i < light_positions.size(); ++i)
        {
            const Vector3d &light_position = light_positions[i];
            const Vector3d &light_color = light_colors[i];
            const Vector3d Li = (light_position - p).normalized();
            
            
            
            Vector3d diff_color = obj_diffuse_color;
            
            const Vector3d diffuse = diff_color * max(Li.dot(N), 0.0);
            
            // Specular contribution

            Vector3d h = (Li - camera_position+ p).normalized();
            const Vector3d specular = obj_specular_color*(pow(fmax(0,h.dot(N)),obj_specular_exponent));
            // Attenuate lights according to the squared distance to the lights
            const Vector3d D = light_position - p;
            lights_color += (diffuse + specular).cwiseProduct(light_color) / D.squaredNorm();
             
        }
        Vector3d C = ambient_light + lights_color;
        
        out.color << C(0),C(1),C(2),1;

        return out;
    };
   

    program.FragmentShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        //TODO: create the correct fragment
        Vector4f color = va.color;
        FragmentAttributes out (color[0], color[1], color[2], color[3]);
        out.position = va.position;
        out.depth = va.position[2];
        return out;
    };
 
    program.BlendingShader = [](const FragmentAttributes &fa, const FrameBufferAttributes &previous) {
        //TODO: implement the depth check
        
        if ( fa.position(2) < previous.depth){
            FrameBufferAttributes out(fa.color[0] * 255, fa.color[1] * 255, fa.color[2] * 255, fa.color[3] * 255);
            out.depth = fa.depth;

            return out;
            
        }
        else{
            return previous;
        }
    };
     
    
}


 
void flat_shading(const double alpha, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{
    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;
    get_shading_program(program);
    Eigen::Matrix4f trafo = compute_rotation(alpha);
    //cout << uniform.projective <<endl;
    //cout << uniform.view <<endl;
    Matrix3d trafo1;

    trafo1 << trafo(0,0), trafo(0,1), trafo(0,2),
                trafo(1,0), trafo(1,1), trafo(1,2),
                trafo(2,0), trafo(2,1), trafo(2,2);
    std::vector<VertexAttributes> vertex_attributes;
    //TODO: compute the normals
    //TODO: set material colors
    
    for (int i = 0; i < facets.rows(); ++i){
        int p1 = facets(i,0);
        int p2 = facets(i,1);
        int p3 = facets(i,2);
        
        
        Vector3d a(vertices(p1,0)-vertices(p2,0),
                   vertices(p1,1)-vertices(p2,1),
                   
                   vertices(p1,2)-vertices(p2,2));
        a << trafo1 * a;
        Vector3d b(vertices(p3,0)-vertices(p2,0),
                   vertices(p3,1)-vertices(p2,1),
                   vertices(p3,2)-vertices(p2,2));
        b << trafo1 * b;
        Vector3d plane = a.cross(b).normalized();
        
        
        VertexAttributes v1 = VertexAttributes(vertices(p1,0),vertices(p1,1),vertices(p1,2));
        VertexAttributes v2 = VertexAttributes(vertices(p2,0),vertices(p2,1),vertices(p2,2));
        VertexAttributes v3 = VertexAttributes(vertices(p3,0),vertices(p3,1),vertices(p3,2));
        v1.trafo = trafo;
        v2.trafo = trafo;
        v3.trafo = trafo;
        v1.normal = plane;
        v2.normal = plane;
        v3.normal = plane;
        vertex_attributes.push_back(v1);
        vertex_attributes.push_back(v2);
        vertex_attributes.push_back(v3);
        
    
    
    }
    rasterize_triangles(program, uniform, vertex_attributes, frameBuffer);
}

void pv_shading(const double alpha, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{
    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;
    get_shading_program(program);

    Eigen::Matrix4f trafo = compute_rotation(alpha);
    Matrix3d trafo1;

    trafo1 << trafo(0,0), trafo(0,1), trafo(0,2),
                trafo(1,0), trafo(1,1), trafo(1,2),
                trafo(2,0), trafo(2,1), trafo(2,2);
    //TODO: compute the vertex normals as vertex normal average
    vector<Vector3d> vect;
    for (int i = 0;i<vertices.rows();++i){
        Vector3d filler;
        vect.push_back(filler);
    }
    
    std::vector<VertexAttributes> vertex_attributes;
    //TODO: create vertex attributes
    //TODO: set material colors
    
    for (int i = 0; i < facets.rows(); ++i){
        int p1 = facets(i,0);
        int p2 = facets(i,1);
        int p3 = facets(i,2);
        
       
        
        Vector3d a(vertices(p1,0)-vertices(p2,0),
                   vertices(p1,1)-vertices(p2,1),
                   vertices(p1,2)-vertices(p2,2));
        a << trafo1 * a;
        Vector3d b(vertices(p3,0)-vertices(p2,0),
                   vertices(p3,1)-vertices(p2,1),
                   vertices(p3,2)-vertices(p2,2));
        b << trafo1 * b;
        Vector3d plane = a.cross(b).normalized();
        vect[p1] = vect[p1] + plane;
        vect[p2] = vect[p2] + plane;
        vect[p3] = vect[p3] + plane;
        
    }
     
    for (int i = 0; i < facets.rows(); ++i){
        int p1 = facets(i,0);
        int p2 = facets(i,1);
        int p3 = facets(i,2);
        VertexAttributes v1 = VertexAttributes(vertices(p1,0),vertices(p1,1),vertices(p1,2));
        VertexAttributes v2 = VertexAttributes(vertices(p2,0),vertices(p2,1),vertices(p2,2));
        VertexAttributes v3 = VertexAttributes(vertices(p3,0),vertices(p3,1),vertices(p3,2));
        
       
        
        v1.normal = vect[p1].normalized();
        v2.normal = vect[p2].normalized();
        v3.normal = vect[p3].normalized();
        v1.trafo = trafo;
        v2.trafo = trafo;
        v3.trafo = trafo;
        
        
        vertex_attributes.push_back(v1);
        vertex_attributes.push_back(v2);
        vertex_attributes.push_back(v3);
    
    
    }
    
    rasterize_triangles(program, uniform, vertex_attributes, frameBuffer);
}

int main(int argc, char *argv[])
{
    setup_scene();

    int W = H * aspect_ratio;
    Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> frameBuffer(W, H);
    vector<uint8_t> image;
    simple_render(frameBuffer);
    
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("simple.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);
    frameBuffer.setConstant(FrameBufferAttributes());
    
    
    wireframe_render(0, frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("wireframe.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);
    frameBuffer.setConstant(FrameBufferAttributes());
    
  
    flat_shading(0, frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("flat_shading.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);
    frameBuffer.setConstant(FrameBufferAttributes());
    
    pv_shading(0, frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("pv_shading.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);
    frameBuffer.setConstant(FrameBufferAttributes());

    //TODO: add the animation
    int delay = 25;

    
    const char *fileName1 = "wireframe.gif";
    vector<uint8_t> image1;
    GifWriter g1;
    GifBegin(&g1,fileName1,frameBuffer.rows(),frameBuffer.cols(),delay);
    for (float i = 0; i < 2*M_PI; i += 0.05)
       {
           frameBuffer.setConstant(FrameBufferAttributes());
           wireframe_render(i,frameBuffer);
           framebuffer_to_uint8(frameBuffer, image1);
           GifWriteFrame(&g1, image1.data(), frameBuffer.rows(), frameBuffer.cols(), delay);
       }

       GifEnd(&g1);
    
    
  
    const char *fileName2 = "flat_shading.gif";
    vector<uint8_t> image2;
    GifWriter g2;
    GifBegin(&g2,fileName2,frameBuffer.rows(),frameBuffer.cols(),delay);
    for (float i = 0; i < 2*M_PI; i += 0.05)
       {
           frameBuffer.setConstant(FrameBufferAttributes());
           flat_shading(i,frameBuffer);
           framebuffer_to_uint8(frameBuffer, image2);
           GifWriteFrame(&g2, image2.data(), frameBuffer.rows(), frameBuffer.cols(), delay);
       }

    GifEnd(&g2);
    
     
    const char *fileName3 = "pv_shading.gif";
    vector<uint8_t> image3;
    GifWriter g3;
    GifBegin(&g3,fileName3,frameBuffer.rows(),frameBuffer.cols(),delay);
    for (float i = 0; i < M_PI; i += 0.2)
       {
           frameBuffer.setConstant(FrameBufferAttributes());
           pv_shading(i,frameBuffer);
           framebuffer_to_uint8(frameBuffer, image3);
           GifWriteFrame(&g3, image3.data(), frameBuffer.rows(), frameBuffer.cols(), delay);
       }

    GifEnd(&g3);
   
    return 0;
}

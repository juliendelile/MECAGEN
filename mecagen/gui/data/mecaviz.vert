
#extension GL_EXT_gpu_shader4 : enable
#extension GL_EXT_texture_integer : enable

 
attribute float scalarR;
attribute float scalarG;
attribute float scalarB;
attribute float hide;

 void main() {

 //vec2 r = vec2(gl_MultiTexCoord0[0],gl_MultiTexCoord0[1]);///4000.0;	//.8
 //gl_TexCoord[0] = gl_MultiTexCoord0;
 //float r = gl_MultiTexCoord0[0];
  
 
 vec4 materialDiffuse;
 vec4 materialAmbient;

	 
	materialDiffuse = vec4(scalarR, scalarG, scalarB,  1.0);
	materialAmbient = vec4(.5*scalarR, .5*scalarG, .5*scalarB,  1.0);
	
	// materialDiffuse = vec4(scalar, sin(3.14*scalar), 1.0 - scalar,  1.0);
	// materialAmbient = vec4(.5*scalar, .5*sin(3.14*scalar), .5*(1.0 - scalar),  1.0);
	


	// materialAmbient = vec4(0.0, 0.0, 0.5 , 1.0);
	// materialDiffuse = vec4(0.0, 0.0, 1.0 , 1.0);
	
	// materialDiffuse = vec4(.40, .40, .0 ,  1.0);
	// materialAmbient = vec4(.40, .40, .0 , 1.0);

vec3 lightpos = vec3(1.0,1.0,1.0);

 vec4 lightDiffuse = vec4(1.0, 1.0, 1.0, 1.0);		/*couleur de la light  blanche*/
 vec4 lightAmbient = vec4(1.0, 1.0, 1.0, 1.0);		//couleur de la light  blanche
 

 vec3 normal;
 vec3 lightDir;

 vec4 diffuse;
 vec4 ambient; 


 float NdotL;

 
 /* first transform the normal into eye space and normalize the result */
 normal = normalize(gl_NormalMatrix * gl_Normal);
 
 /* now normalize the light's direction. Note that according to the
OpenGL specification, the light is stored in eye space. Also since 
we're talking about a directional light, the position field is actually 
direction */
lightDir = normalize(vec3(lightpos));

 
 
 
 
 /* compute the cos of the angle between the normal and lights direction. 
The light is directional so the direction is constant for every vertex.
Since these two are normalized the cosine is the dot product. We also 
need to clamp the result to the [0,1] range. */
 NdotL = max(dot(normal, lightDir), 0.0);

 /* Compute the diffuse term */
 diffuse = materialDiffuse * lightDiffuse;
 
 /* Compute the ambient and globalAmbient terms */
 ambient = materialAmbient * lightAmbient;
 //globalAmbient = lightAmbient * materialAmbient;

 gl_FrontColor =  NdotL * diffuse + ambient;// + globalAmbient;
 //gl_FrontColor =   ambient;// + globalAmbient;

 
	gl_Position = ftransform();


if(hide > .0){

	gl_Position = vec4(-10.0,10.0,10.0,10.0);
}


 //clipping.. code bizarre mais qui fonctionne
 vec4 vPosEyeSpace = gl_ModelViewMatrix * gl_Vertex;
	gl_ClipVertex = vPosEyeSpace;
 
 
 
 
 }

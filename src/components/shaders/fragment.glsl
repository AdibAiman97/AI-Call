uniform vec2 u_resolution;
uniform vec3 u_primaryColor;
uniform float u_time;

void main(){
  vec2 st=gl_FragCoord.xy/u_resolution;// Normalized screen coordinates
  
  // Create animated radial gradient with multiple colors
  vec2 center = vec2(0.5, 0.5);
  float dist = distance(st, center);
  
  // Create color variations for gradient
  vec3 color1 = u_primaryColor;                           // Base color
  vec3 color2 = u_primaryColor * 1.5;                     // Brighter version
  vec3 color3 = u_primaryColor * 0.6 + vec3(0.3, 0.3, 0.3); // Darker with some white
  
  // Animated gradient based on distance and time
  float gradient1 = smoothstep(0.0, 0.3, dist);
  float gradient2 = smoothstep(0.2, 0.6, dist);
  float gradient3 = smoothstep(0.4, 1.0, dist);
  
  // Time-based color shifting for animation
  float timeShift = sin(u_time * 2.0) * 0.3 + 0.7;
  
  // Mix colors based on gradients
  vec3 finalColor = mix(color1, color2, gradient1 * timeShift);
  finalColor = mix(finalColor, color3, gradient2);
  
  // Add some dynamic variation
  float noise = sin(st.x * 10.0 + u_time) * sin(st.y * 10.0 + u_time * 0.7) * 0.1 + 0.9;
  finalColor *= noise;
  
  // Add subtle pulsing effect
  float pulse = sin(u_time * 3.0) * 0.2 + 0.8;
  finalColor *= pulse;
  
  gl_FragColor=vec4(finalColor, 1.0);
}
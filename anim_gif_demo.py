'''
generate cool animation morphing effect for all digits
sampler is assumed to be an instance of Sampler()
'''

def create_z_array():
  z_array = []
  for i in range(10):
    z_array.append(sampler.encode(sampler.get_random_specific_mnist(i)))
    sampler.show_image_from_z(z_array[i])
  return z_array

def make_img_data_array(z_array_input, sinusoid = True, fps=24, x_dim = 1080, y_dim = 1080):
  z_array = list(z_array_input)
  n = len(z_array)
  data = []
  for i in range(n-1):
    print "Morphing Image #", i
    sampler.show_image_from_z(z_array[i])
    data.append(sampler.morph(z_array[i], z_array[i+1], fps, sinusoid = sinusoid, x_dim = x_dim, y_dim = y_dim))
  sampler.show_image_from_z(z_array[n-1])
  data.append(sampler.morph(z_array[n-1], z_array[0], fps, sinusoid = sinusoid, x_dim = x_dim, y_dim = y_dim))
  return data

def write_data_array_as_gif(data_array, filename, fps = 24):
  result = []
  for i in range(len(data_array)):
    result = result + data_array[i]
  sampler.save_anim_gif(result, filename, 1.0 / float(fps))


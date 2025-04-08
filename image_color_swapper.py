import PIL.Image as Image

image_path = '../van_zeland_2.png'

img = Image.open(image_path).convert("RGBA")

# new stuff

width, height = img.size

pixels = img.load()

for x in range(0, width):
    for y in range(0, height):
        if pixels[x,y] == (159, 0, 0, 255):
            pixels[x,y] = (255, 255, 255, 255)

for x in range(12*width//30, width//2):
    for y in range(height//2, height):
        if (pixels[x,y][0] > pixels[x,y][1]) and (pixels[x,y][0] > pixels[x,y][2]):
            pixels[x,y] = (255, 255, 255, 255) 



# # Define the color to replace and the replacement color
# color_to_replace = (159, 0, 0, 255)  # darker red Red
# replacement_color = (255, 255, 255, 255)  # white with full opacity

# # Create a new image with replaced colors
# data = img.getdata()
# new_data = [
#     replacement_color if pixel == color_to_replace else pixel
#     for pixel in data
# ]

# # Update the image data
# img.putdata(new_data)

# # Save the updated image
output_path = "output.png"  # Replace with your desired output path
img.save(output_path)

print(f"Color replaced and saved to {output_path}")
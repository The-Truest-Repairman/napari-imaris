from imaris_ims_file_reader.ims import ims

# Replace with the path to your .ims file
ims_path = "../images/BP003_2-3.ims"

# Load the .ims file
image_data = ims(ims_path)

# Access a specific slice (e.g., timepoint 0, channel 0, z-layer 5)
slice_data = image_data[0, 0, 5, :, :]

# Print the shape of the slice
print("Slice shape:", slice_data.shape)
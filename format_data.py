import sys
import os
from imutils import paths
import csv


def main():
    formatted_data = {}
    with open(sys.argv[1], 'w', newline='') as f:
        writer = csv.writer(f)
        try:
            for file in sys.argv[2:]:
                images = paths.list_images(file)
                for image in images:
                    raw_image_name = os.path.basename(image)
                    temp_val = [0, 0, 0]  # Depth, Above, Below
                    data_values = raw_image_name.split("-")  # Above/Below Blur-Value ImageIDNum Clarity-Value DateTime
                    key_name = data_values[2] + data_values[4] + "-" + data_values[3] + ""
                    if key_name in formatted_data:
                        if data_values[0] == "below":
                            formatted_data[key_name][2] = data_values[1]
                        elif data_values[0] == "above":
                            formatted_data[key_name][1] = data_values[1]
                    else:
                        temp_val[0] = data_values[3]
                        if data_values[0] == "below":
                            temp_val[2] = data_values[1]
                        elif data_values[0] == "above":
                            temp_val[1] = data_values[1]
                        formatted_data[key_name] = temp_val
                    
            for value in formatted_data:
                temp_list = [value] + formatted_data[value]
                writer.writerow(temp_list)
        except csv.Error as e:
            sys.exit('rip')


if __name__ == "__main__":  # CSV Name, File Path
    main()

import pandas as pd
import xml.etree.ElementTree as ET

# Your XML annotation (replace with the actual content)
xml_annotation = """
<annotation>
<filename>000000000.tif</filename>
<source>
<annotation>ESRI ArcGIS Pro</annotation>
</source>
<size>
<width>256</width>
<height>256</height>
<depth>3</depth>
</size>
<object>
<name>50</name>
<bndbox>
<xmin>67.52</xmin>
<ymin>105.44</ymin>
<xmax>147.52</xmax>
<ymax>185.44</ymax>
</bndbox>
</object>
<object>
<name>11</name>
<bndbox>
<xmin>181.20</xmin>
<ymin>15.89</ymin>
<xmax>256.00</xmax>
<ymax>95.89</ymax>
</bndbox>
</object>
</annotation>
"""

# Parse the XML
root = ET.fromstring(xml_annotation)

# Initialize lists to store data
name_list = []
xmin_list = []
xmax_list = []
ymin_list = []
ymax_list = []

# Extract bounding box coordinates
for obj in root.findall(".//object"):
    name = float(obj.find("name").text)
    xmin = float(obj.find("bndbox/xmin").text)
    xmax = float(obj.find("bndbox/xmax").text)
    ymin = float(obj.find("bndbox/ymin").text)
    ymax = float(obj.find("bndbox/ymax").text)

    name_list.append(name)
    xmin_list.append(xmin)
    xmax_list.append(xmax)
    ymin_list.append(ymin)
    ymax_list.append(ymax)

# Create a DataFrame
df = pd.DataFrame({
    "name": name_list,
    "xmin": xmin_list,
    "xmax": xmax_list,
    "ymin": ymin_list,
    "ymax": ymax_list
})

# Save to CSV
df.to_csv("bounding_boxes.csv", index=False)

print("CSV file 'bounding_boxes.csv' created successfully!")

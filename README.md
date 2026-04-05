# Mining Equipment Activity Classifier

## 1. Project Overview
This system uses Computer Vision to track Excavator and Truck utilization in real-time. It classifies activities and streams data to a PostgreSQL database.

## 2. The Articulated Equipment Challenge (How I solved it)
Standard bounding boxes fail when an excavator is stationary but its arm is moving. 
- **The Solution:** I implemented Instance Segmentation to track the **Bucket** as a separate entity.
- **Arm-Only Motion:** By calculating the Euclidean distance of the bucket centroid across 15-frame intervals, the system detects movement even if the main chassis (the body of the excavator) is still.
- **Activity Logic:** - **Digging:** Bucket is below the horizontal mid-line of the excavator.
    - **Dumping:** Bucket centroid enters the "Tight Zone" (100px padding) of a detected truck.

## 3. Data Pipeline
- **Vision:** YOLOv8 (Custom Segment Model + COCO).
- **Dashboard:** Streamlit UI.
- **Storage:** PostgreSQL (Default 'postgres' DB).
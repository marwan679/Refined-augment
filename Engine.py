import cv2
import numpy as np
import os

def apply_face_augmentation(camera_index: int | str = 0, overlay_path: str = 'AR_photo.png', cascade_path: str = 'haarcascade_frontalface_default.xml'):
    """
    Captures video, detects faces, and overlays an AR image above the detected face.
    """
    # 1. Initialize Resources
    cap = cv2.VideoCapture(camera_index)
    
    # Ensure paths are handled as absolute to prevent read errors
    face_cascade = cv2.CascadeClassifier(os.path.abspath(cascade_path))
    overlay_img = cv2.imread(os.path.abspath(overlay_path))

    if overlay_img is None:
        print(f"Error: Could not load overlay image at {os.path.abspath(overlay_path)}")
        return

    if face_cascade.empty():
        print(f"Error: Could not load Haar Cascade at {os.path.abspath(cascade_path)}")
        return

    print("AR Session Started. Press 'q' to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Standardize orientation
            frame = cv2.flip(frame, 1)
            img_aug = frame.copy()
            gray = cv2.cvtColor(img_aug, cv2.COLOR_BGR2GRAY)
            
            # 2. Face Detection
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                # Define geometry for the AR overlay (positioned exactly one face-height above the face)
                # Note: We use np.float32 for homography calculation
                src_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
                dst_points = np.array([
                    [x, y - h], [x + w, y - h], [x + w, y], [x, y]
                ], dtype=np.float32)

                # Resize the AR image to fit the face width/height
                ar_object = cv2.resize(overlay_img, (w, h))

                # Compute the Homography matrix to map the sticker to the face coordinates
                matrix, _ = cv2.findHomography(src_points, dst_points)
                
                # Apply the perspective warp to the sticker
                img_warp = cv2.warpPerspective(ar_object, matrix, (frame.shape[1], frame.shape[0]))

                # 3. Masking and Blending
                # Create a black mask, then fill the area where the sticker will go with white
                mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
                cv2.fillPoly(mask, [np.int32(dst_points)], 255)
                mask_inv = cv2.bitwise_not(mask)

                # Use the masks to blend the sticker into the original frame
                img_aug = cv2.bitwise_and(img_aug, img_aug, mask=mask_inv)
                img_aug = cv2.bitwise_or(img_warp, img_aug)

            # 4. Display the result
            cv2.imshow('AR Face Overlay', img_aug)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # 5. Cleanup Resources
        cap.release()
        cv2.destroyAllWindows()
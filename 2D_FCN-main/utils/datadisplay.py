import cv2
import numpy as np

#shows a random video with its mask
def show_data(X, Y_true, image_shape, Y_pred = None):
    #de-one-hot
    Y_true = np.array(Y_true.argmax(axis=2) * 255,dtype=np.uint8)
    #random show a pair of video & mask
    n = np.random.randint(X.shape[0])
    images,msk = X[n],Y_true[n]
    for i,image in enumerate(images):
        cv2.putText(image,
                    str(i),
                    (10, image.shape[0]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        cv2.imshow("images",image)
        if i == len(images) - 1:
            cv2.imshow("true_mask", msk.reshape(image_shape[0:2]))
            if Y_pred is not None:
                Y_pred = np.array(Y_pred.argmax(axis=2) * 255, dtype=np.uint8)
                msk_pred = Y_pred[n]
                cv2.imshow("pred_mask", msk_pred.reshape(image_shape[0:2]))
        cv2.waitKey(0)
    cv2.destroyAllWindows()
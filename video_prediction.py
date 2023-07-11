import numpy as np
import cv2    
import matplotlib.pyplot as plt
import os


def ExtractFrames(cfg, start=0, batch=0):
    #Load video path
    def resize(im):
        return cv2.resize(im, dsize=(cfg['img_width'], cfg['img_height']), interpolation=cv2.INTER_NEAREST)

    video_path = os.path.join(cfg['video_dataset'], cfg['video_name']  + cfg['form_video'])

    capture = cv2.VideoCapture(video_path) #read the video from the path
    frame_width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)//2)
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)//2)
    
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if batch == 0:
        batch = length
    video = np.zeros((batch,cfg['img_height'], cfg['img_width'],3), dtype=np.uint8)
   
    for n in range(batch):
        ret, frame = capture.read() #read the video frame by frame
        frame = frame[...,::-1]
        video[n] = resize(frame)

    capture.release()
    return video, [frame_width, frame_height],  fps


import io
def get_img_from_fig(fig, dpi=60):

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)

    return img

# helper function for data visualization (with overlay)
def visualize_pred(image, pred_mask, cfg, frame_num, i=0):
    image = (image*255).astype(np.uint8)
    image     = cv2.resize(image, dsize=(cfg['video_shape'][0], cfg['video_shape'][1]), interpolation=cv2.INTER_NEAREST)
    pred_mask = cv2.resize(pred_mask*1.0, dsize=(cfg['video_shape'][0], cfg['video_shape'][1]), interpolation=cv2.INTER_NEAREST)

    """PLot image, mask, prediction in one row."""
    f = plt.figure(figsize=(20, 5))

    #Plot original image
    plt.subplot(1,3,1)
    plt.axis('off')
    plt.title('Original Image (Frame ' + frame_num + ')')
    plt.imshow(image)

    #Plot predicted mask
    plt.subplot(1,3,2)
    plt.axis('off')
    plt.title('Predicted mask')
    plt.imshow(pred_mask)

    # Overlay the predicted mask on the original image
    pred_mask[pred_mask<=0.5] = 0.3
    pred_mask[pred_mask> 0.5]  = 1
    plt.subplot(1,3,3)
    plt.axis('off')
    plt.title('Original Image and Predicted Mask Overlayed')
    plt.imshow((image*pred_mask[...,None]).astype(np.uint8))

    img = get_img_from_fig(f)
    plt.close('all')

    return img

def VideoPred (model, cfg):
    #Load model and frames dataset
    video_np, fr_shape, fps = ExtractFrames(cfg, 0, 0)
    video_np = video_np.astype(np.float32)/255.0
    cfg['video_shape'] = fr_shape


    mask_pred = model.predict(video_np)
    mask_pred = mask_pred>0.5


    # Reconstruct the video from the predicted frames

    save_video = os.path.join(cfg['output'], 'video_output', cfg['video_name'] + '_output.mp4')


    frame = visualize_pred(video_np[0], mask_pred[0], cfg, 1)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_video, fourcc, fps, (frame.shape[1], frame.shape[0]))


    for i in range(len(video_np)):
        frame_num = i + 1  # Frame numbers start from 1
        print(f'frame {frame_num:04d}')
        frame = visualize_pred(video_np[i], mask_pred[i], cfg, frame_num)
        video.write(frame)
        
    video.release()


   
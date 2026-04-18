import os
import cv2
import math
import torch
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils
from patchify import patchify, unpatchify

from utils.constants import Constants
from utils.plot import visualize
from utils.logger import custom_logger
from utils.root_config import get_root_config


################################# Prompt Filtering Helper Functions #################################

# Maps menu number → (internal class name, display name)
PROMPT_MENU_MAP = {
    '1': ('background', 'Land'),
    '2': ('building',   'Building'),
    '3': ('woodland',   'Vegetation'),
    '4': ('water',      'Water'),
}


def ask_yes_no(question):
    """Ask a Yes/No question and return True for Yes, False for No."""
    while True:
        answer = input(f"\n{question} (Yes/No): ").strip().lower()
        if answer in ('yes', 'y'):
            return True
        elif answer in ('no', 'n'):
            return False
        else:
            print("Invalid input. Please enter Yes or No.")


def prompt_class_selection_menu(all_classes_from_config):
    """
    Show interactive class selection menu.
    Returns:
        list  → selected class names (when user presses 5 with at least one class added)
        None  → user pressed 6 (Exit / cancel prompt filtering → fall back to all config classes)
    """
    selected_classes = []

    while True:
        print("\n1. Land\n2. Building\n3. Vegetation\n4. Water\n5. Done\n6. Exit")
        choice = input("Enter your choice: ").strip()

        if choice in PROMPT_MENU_MAP:
            cls_name, display_name = PROMPT_MENU_MAP[choice]
            # Only add if the class exists in the dataset's all_classes list
            if cls_name in all_classes_from_config:
                if cls_name not in selected_classes:
                    selected_classes.append(cls_name)
                    print(f"{display_name} Class is added...")
                else:
                    print(f"{display_name} Class is already added.")
            else:
                print(f"{display_name} class is not available in the current dataset configuration.")

        elif choice == '5':
            if not selected_classes:
                print("No Classes were added.")
                # Do not exit menu — let user add classes or press 6 to cancel
            else:
                return selected_classes

        elif choice == '6':
            # Cancel prompt filtering → caller will fall back to all config classes
            return None

        else:
            print("Invalid input. Please enter a number between 1 and 6.")

#####################################################################################################


if __name__ == "__main__":

    ################################# Loading Variables and Paths from Config #################################

    ROOT, slice_config = get_root_config(__file__, Constants)

    # get the required variable values from config
    log_level = slice_config['vars']['log_level']
    file_type = slice_config['vars']['file_type']
    patch_size = slice_config['vars']['patch_size']  # size of each patch and window
    encoder = slice_config['vars']['encoder']        # the backbone/encoder of the model
    encoder_weights = slice_config['vars']['encoder_weights']
    classes = slice_config['vars']['test_classes']   # default classes from config (unchanged)
    device = slice_config['vars']['device']
    all_classes_config = slice_config['vars']['all_classes']  # all classes in dataset

    # get the log file dir from config
    log_dir = ROOT / slice_config['dirs']['log_dir']
    # make the directory if it does not exist
    log_dir.mkdir(parents = True, exist_ok = True)
    # get the log file path
    log_path = log_dir / slice_config['vars']['test_log_name']
    # convert the path to string in a format compliant with the current OS
    log_path = log_path.as_posix()

    # initialize the logger
    logger = custom_logger("Land Cover Semantic Segmentation Test Logs", log_path, log_level)

    # get the dir of input images for inference from config
    img_dir = ROOT / slice_config['dirs']['data_dir'] / slice_config['dirs']['test_dir'] / slice_config['dirs']['image_dir']
    img_dir = img_dir.as_posix()

    # get the dir of input masks for inference from config
    gt_mask_dir = ROOT / slice_config['dirs']['data_dir'] / slice_config['dirs']['test_dir'] / slice_config['dirs']['mask_dir']
    gt_mask_dir = gt_mask_dir.as_posix()

    # get the model path from config
    model_name = slice_config['vars']['model_name']
    model_path = ROOT / slice_config['dirs']['model_dir'] / model_name
    model_path = model_path.as_posix()

    # get the predicted masks dir from config
    pred_mask_dir = ROOT / slice_config['dirs']['output_dir'] / slice_config['dirs']['pred_mask_dir']
    # make the directory if it does not exist
    pred_mask_dir.mkdir(parents = True, exist_ok = True)
    pred_mask_dir = pred_mask_dir.as_posix()

    # get the prediction plots dir from config
    pred_plot_dir = ROOT / slice_config['dirs']['output_dir'] / slice_config['dirs']['pred_plot_dir']
    # make the directory if it does not exist
    pred_plot_dir.mkdir(parents = True, exist_ok = True)
    pred_plot_dir = pred_plot_dir.as_posix()

    ###########################################################################################################

    ####################################### Functional Part of Program ########################################

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
    model = torch.load(model_path, map_location=torch.device(device))

    # Default class values from config (used when prompt filtering is OFF)
    default_class_values = [Constants.CLASSES.value.index(cls.lower()) for cls in classes]

    img_list = list(filter(lambda x: x.endswith((file_type)), os.listdir(img_dir)))

    print(f"\nTotal images found to test: {len(img_list)}")
    logger.info(f"Total images found to test: {len(img_list)}")

    # ───────────────────────────────────────────────
    # STEP 1 — Ask initial Prompt Filtering question
    # ───────────────────────────────────────────────
    enable_prompt = ask_yes_no("Do you want to Enable Prompt Filtering?")
    if enable_prompt:
        selected = prompt_class_selection_menu(all_classes_config)

        if selected is None:
            print("\nExiting program...")
            exit()   # 🔥 FULL EXIT

        else:
            current_classes      = selected
            current_class_values = [
                Constants.CLASSES.value.index(cls.lower())
                for cls in current_classes
        ]
    else:
        # No prompt filtering — use all classes from config
        current_classes      = classes
        current_class_values = default_class_values

    # ───────────────────────────────────────────────
    # STEP 2 — Process images
    # ───────────────────────────────────────────────
    exit_program = False
    try:
        for img_index, filename in enumerate(img_list):

            print(f"\nPreparing image file {filename}...")
            logger.info(f"Preparing image file {filename}...")

            # reading image
            try:
                image = cv2.imread(os.path.join(img_dir, filename), 1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.error(f"Could not read image file {filename}!")
                raise e

            # # reading ground truth mask
            # try:
            #     gt_mask = cv2.imread(os.path.join(gt_mask_dir, filename), 0)
            #     # filter classes
            #     gt_masks = [(gt_mask == v) for v in current_class_values]
            #     gt_mask = np.stack(gt_masks, axis=-1).astype('float')
            #     gt_mask = gt_mask.argmax(2)
            # except Exception as e:
            #     logger.error(f"Could not read ground truth mask file {filename}!")
            #     raise e

            # padding image to be perfectly divisible by patch_size
            try:
                pad_height = (math.ceil(image.shape[0] / patch_size) * patch_size) - image.shape[0]
                pad_width  = (math.ceil(image.shape[1] / patch_size) * patch_size) - image.shape[1]
                padded_shape = ((0, pad_height), (0, pad_width), (0, 0))
                image_padded = np.pad(image, padded_shape, mode='reflect')
            except Exception as e:
                logger.error("Could not pad the image!")
                raise e

            # dividing image into patches according to patch_size in overlapping mode
            try:
                patches      = patchify(image_padded, (patch_size, patch_size, 3), step=patch_size//2)[:, :, 0, :, :, :]
                mask_patches = np.empty(patches.shape[:-1], dtype=patches.dtype)
            except Exception as e:
                logger.error("Could not patchify the image!")
                raise e

            print("\nImage preparation done successfully!")
            logger.info("Image preparation done successfully!")

            # model prediction
            # 🔁 LOOP FOR SAME IMAGE
            while True:

                print(f"\nProcessing {filename} with classes: {current_classes}")

                mask_patches = np.empty(patches.shape[:-1], dtype=patches.dtype)

                # prediction
                for i in tqdm(range(0, patches.shape[0])):
                    for j in range(0, patches.shape[1]):
                        img_patch  = preprocessing_fn(patches[i, j])
                        img_patch  = img_patch.transpose(2, 0, 1).astype('float32')
                        x_tensor   = torch.from_numpy(img_patch).to(device).unsqueeze(0)

                        pred_mask  = model.predict(x_tensor)
                        pred_mask  = pred_mask.squeeze().cpu().numpy().round()
                        pred_mask  = pred_mask.transpose(1, 2, 0)
                        pred_mask  = pred_mask.argmax(2)

                        mask_patches[i, j] = pred_mask

                # reconstruct
                pred_mask = unpatchify(mask_patches, image_padded.shape[:-1])
                pred_mask = pred_mask[:image.shape[0], :image.shape[1]]

    #             # filter classes
    #             if len(current_class_values) == 1:
    # # ✅ Single class → binary mask
    #                 pred_mask = (pred_mask == current_class_values[0]).astype(np.uint8) * 255
    #             else:
    # # ✅ Multi-class → normal behavior
    #                 pred_masks = [(pred_mask == v) for v in current_class_values]
    #                 pred_mask  = np.stack(pred_masks, axis=-1).astype('float')
    #                 pred_mask  = pred_mask.argmax(2)

    # filter classes
                if len(current_class_values) == 1:
                    # Single class → binary mask
                    pred_mask = (pred_mask == current_class_values[0]).astype(np.uint8) * 255
                else:
                    # Multi-class → only keep pixels belonging to selected classes
                    valid_mask = np.zeros_like(pred_mask, dtype=bool)
                    for v in current_class_values:
                        valid_mask |= (pred_mask == v)

                    pred_masks = [(pred_mask == v) for v in current_class_values]
                    filtered   = np.stack(pred_masks, axis=-1).astype('float')
                    filtered   = filtered.argmax(2) + 1  # 1-indexed; 0 = not selected

                    pred_mask  = np.where(valid_mask, filtered, 0).astype(np.uint8)

                    # Scale values for visible output (0→0, 1→85, 2→170, 3→255 for up to 3 classes)
                    scale = 255 // len(current_class_values)
                    pred_mask = (pred_mask * scale).astype(np.uint8)

                print(f"Classes present after filtering: {current_classes}")

                # save mask
                cv2.imwrite(os.path.join(pred_mask_dir, filename), pred_mask)

                # save plot
                plot_fig = visualize(
                    image=image,
                    predicted_mask=pred_mask
                )
                plot_fig.savefig(os.path.join(pred_plot_dir, filename.split('.')[0] + '.png'))

                print("Prediction done and saved!")

                # 🔁 ASK FOR SAME IMAGE AGAIN
                again = ask_yes_no("Do you want to apply different Prompt Filtering for THIS image?")

                if again == False:
                    break

                # select new classes
                selected = prompt_class_selection_menu(all_classes_config)

                if selected is None:
                    print("\nExiting prompt filtering for this image...")
                    exit_program = True
                    break
                else:
                    current_classes = selected
                    current_class_values = [
                        Constants.CLASSES.value.index(cls.lower())
                        for cls in current_classes
                    ]
            if exit_program:
                break
    except Exception as e:
        logger.error("No images found in 'data/test/images' folder!")
        raise e

    ###########################################################################################################
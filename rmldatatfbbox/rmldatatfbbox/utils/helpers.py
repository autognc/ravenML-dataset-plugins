import os
import shutil
import pandas as pd
import json
import cv2
from ravenml.utils.question import cli_spinner
from rmldatatfbbox.utils.io_utils import copy_data_locally, download_data_from_s3
from rmldatatfbbox.utils.classes import METADATA_PREFIX, BoundingBox, BBoxLabeledImage

### FUNCTIONS ###
def default_filter_and_load(data_source, **kwargs):
    cli_spinner("Loading metadata...", ingest_metadata, data_source, kwargs)

    tags_df = load_metadata()
    filter_metadata = {"groups": []}
    
    # ask the user if they would like to perform filtering
    # if yes, enter a loop that supplies filter options
    # if no, skip
    """if user_confirms(
            message="Would you like to filter out any of the data ({} images total)?".format(len(tags_df)),
            default=False):
        sets = {}
        # outer loop to determine how many sets the user will create
        try:
            while True:
                subset = tags_df
                this_group_filters = []
                len_subsets = [len(subset)]
                # inner loop to handle filtering for ONE set
                while True:

                    # if filters have been applied, display them to the user
                    # to help guide their next choice
                    if len(this_group_filters) > 0:
                        filters_applied = [
                            "   > " + (" " + f["type"] + " ").join(f["tags"]) +
                            " ({} -> {})".format(len_subsets[i],
                                                len_subsets[i + 1])
                            for i, f in enumerate(this_group_filters)
                        ]
                        print(Fore.MAGENTA + "ℹ Filters already applied:\n{}".
                            format("\n".join(filters_applied)))

                    selected_tags = user_selection(
                        message=
                        "Please select a set of tags with which to apply a filter:",
                        choices=list(tags_df),
                        selection_type="checkbox")
                    filter_type = user_selection(
                        message=
                        "Which filter would you like to apply to the above set?",
                        choices=["AND (intersection)", "OR (union)"],
                        selection_type="list")

                    if filter_type == "AND (intersection)":
                        subset = and_filter(subset, selected_tags)
                        this_group_filters.append({
                            "type": "AND",
                            "tags": selected_tags
                        })
                    elif filter_type == "OR (union)":
                        subset = or_filter(subset, selected_tags)
                        this_group_filters.append({
                            "type": "OR",
                            "tags": selected_tags
                        })
                    print(
                        Fore.GREEN +
                        "ℹ There are {} images that meet the filter criteria selected."
                        .format(len(subset)))
                    len_subsets.append(len(subset))

                    if not user_confirms(
                            message=
                            "Would you like to continue filtering this set?",
                            default=False):
                        set_name = user_input(
                            message="What would you like to name this set?",
                            validator=FilenameValidator)
                        sets[set_name] = subset
                        filter_metadata["groups"].append({
                            "name":
                            set_name,
                            "filters":
                            this_group_filters
                        })
                        break
                    
                if not user_confirms(
                        message=
                        "Would you like to create more sets via filtering?",
                        default=False):
                    break

            sets_to_join = []
            for set_name, set_data in sets.items():
                how_many = user_input(
                    message=
                    'How many images of set "{}" would you like to use? (?/{})'
                    .format(set_name, len(set_data)),
                    validator=IntegerValidator,
                    default=str(len(set_data)))
                n = int(how_many)
                sets_to_join.append(
                    set_data.sample(n, replace=False, random_state=42))

                # find the right group within the metadata dict and add the number
                # included to it
                for group in filter_metadata["groups"]:
                    if group["name"] == set_name:
                        group["number_included"] = n

            image_ids = join_sets(sets_to_join).index.tolist()

        except Exception as e:
            print(e)
            sys.exit(1)
    else: """
    image_ids = tags_df.index.tolist()

    # condition function for S3 download and local copying
    def need_file(filename):
        if len(filename) > 0:
            image_id = filename[filename.index('_')+1:filename.index('.')]
            if image_id in image_ids:
                return True

    if(data_source == "Local"):
        cli_spinner("Copying data locally...", copy_data_locally, source_dir=kwargs["data_filepath"], 
                    condition_func=need_file)
    elif(data_source == "S3"):
        cli_spinner("Downloading data from S3...", download_data_from_s3, bucket_name=kwargs["bucket"], 
                    filter_vals=kwargs['filter_vals'], condition_func=need_file)

    # sequester data for this specific run    
    cwd = os.getcwd() # Used to be Path.cwd
    temp_dir = cwd + '/data/temp'
    data_dir = cwd + '/data'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    cli_spinner("Copying data into temp folder...", copy_data_locally,
        source_dir=data_dir, dest_dir=temp_dir, condition_func=need_file)

    return image_ids, filter_metadata, temp_dir

def ingest_metadata(data_source, kwargs):
    only_json_func = lambda filename: filename.startswith("meta_")

    if data_source == "Local":
        copy_data_locally(
            source_dir=kwargs["data_filepath"], condition_func=only_json_func)
    elif data_source == "S3":
        download_data_from_s3(
            bucket_name=kwargs["bucket"],
            filter_vals=kwargs["filter_vals"],
            condition_func=only_json_func)

def load_metadata():
    """Loads all image metadata JSONs and loads their tags

    Returns:
        DataFrame: a pandas DataFrame storing image IDs and associated tags;
            index (rows) = image ID (str)
            column headers = the tags themselves
            columns = True/False values for whether the image has the tag in
                in that column header
    """
    tags_df = pd.DataFrame()
    cwd = os.getcwd() # Used to be Path.cwd()
    data_dir = cwd + '/data'

    for dir_entry in os.scandir(data_dir):
        if not dir_entry.name.startswith(METADATA_PREFIX):
            continue
        image_id = dir_entry.name.replace(METADATA_PREFIX, '').replace(".json", '')
        with open(dir_entry.path, "r") as read_file:
            data = json.load(read_file)
        tag_list = data.get("tags", ['untagged'])
        if len(tag_list) == 0:
            tag_list = ['untagged']
        temp = pd.DataFrame(
            dict(zip(tag_list, [True] * len(tag_list))), index=[image_id])
        tags_df = pd.concat((tags_df, temp), sort=False)
    tags_df = tags_df.fillna(False)

    return tags_df

def construct_all(image_ids, **kwargs):
    labeled_images = {}

    for image_id in image_ids:
        labeled_images[image_id] = construct(image_id, **kwargs)
    
    return labeled_images

def construct(image_id, **kwargs):
    temp_dir = kwargs['temp_dir']
    if temp_dir is None:
        cwd = os.getcwd() # Used to be Path.cwd()
        data_dir = cwd + '/data'
    else:
        data_dir = temp_dir

    image_filepath = None
    image_type = None
    file_extensions = [".png", ".jpg", ".jpeg"]
    for extension in file_extensions:
        if os.path.exists(data_dir + '/' + f'image_{image_id}{extension}'):
            image_filepath = data_dir + '/' + f'image_{image_id}{extension}'
            image_type = extension
            ydim, xdim = tuple(cv2.imread(image_filepath).shape[:2])
            break

    if image_filepath is None:
        raise ValueError("Hmm, there doesn't seem to be a valid image filepath.")

    with open(data_dir + '/' + f"meta_{image_id}.json", "r") as f:
        meta = json.load(f)

    label_boxes = []
    for label, b in meta['bboxes'].items():
        label_boxes.append(BoundingBox(label, b['xmin'], b['xmax'], b['ymin'], b['ymax']))
        add_label_int(label)

    bbox = BBoxLabeledImage(image_id, image_filepath, image_type, label_boxes, xdim, ydim)

    return bbox

def add_label_int(label_to_add):
    if label_to_add not in BBoxLabeledImage._label_to_int_dict.keys():
        # add the new label
        BBoxLabeledImage._label_to_int_dict[label_to_add] = None

        # renumber all values
        BBoxLabeledImage.renumber_label_to_int_dict()

def write_label_map(dataset_name):
        """Writes out the TensorFlow Object Detection Label Map
        
        Args:
            dataset_name (str): the name of the dataset
        """
        dataset_path = os.getcwd() + '/dataset/' + dataset_name
        label_map_filepath = dataset_path + '/label_map.pbtxt'
        label_map = []
        for label_name, label_int in BBoxLabeledImage._label_to_int_dict.items():
            label_info = "\n".join([
                "item {", "  id: {id}".format(id=label_int),
                "  name: '{name}'".format(name=label_name), "}"
            ])
            label_map.append(label_info)
        with open(label_map_filepath, 'w') as outfile:
            outfile.write("\n\n".join(label_map))

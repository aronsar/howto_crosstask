import pickle
import utils

YT_META = "data/YT_meta.pkl"
YT_OLD_META = "data/YT_embeds.pkl"

def create_yt_meta():
    if not os.path.isfile(YT_OLD_META):
        print("Error: Cannot find {} when creating YT_meta.pkl".format(YT_OLD_META))
        raise FileNotFoundError

    # YT Dict [task][video_id]
    with open(YT_META, "rb") as f:
        YT_embeds = pickle.load(f)

    videos_meta = {}
    for task_id, vids in YT_embeds.items():
        videos_meta[task_id] = {}
        for video_id, (yt_title, _) in vids.items():
            title = yt_title.rstrip('\n')
            if title == "":
                continue     
            videos_meta[task_id][video_id] = title

    pickle.dump(videos_meta, open(YT_META, "wb"))

def check_yt_meta():
    with open(YT_META, "rb") as f:
        yt_meta = pickle.load(f)
    
    task_num = 0
    row_count = 0
    for task_id, vids in yt_meta.items():
        task_num += 1
        for video_id, title in vids.items():
            print("Task ID: {}, Video ID: {}, Title {}".format(task_id, video_id, title))
            row_count += 1

    print("Task Number: {}, Row Number {}".format(task_num, row_count))

def get_yt_titles():
    with open(YT_META, "rb") as f:
        yt_meta = pickle.load(f)
    
    task_steps , _, _ = utils.load_task_steps()
    primary_task_ids = task_steps.keys()

    titles = []
    for task_id, vids in yt_meta.items():
        if task_id in primary_task_ids:
            for video_id, title in vids.items():
                titles.append(title)
    
    print("Retrieve YT Title Count: {}".format(len(titles)))
    return titles
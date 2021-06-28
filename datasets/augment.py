import io
import pandas as pd
from PIL import Image
from pathlib import Path
import os
import pathlib
import tqdm

i = 0
toplevelfolder = "datasets"
starting_point = Path(__file__).absolute()
if starting_point.parents:
    # print(list(starting_point.parents))
    while (starting_point.parents[i].name != toplevelfolder):  #finds the abspath no matter the location
        i += 1
abspath = starting_point.parents[i].absolute()
# print(abspath)

biasedcsv = 'datasets\\data\\biased_dataset.csv'
debiasedcsv = 'datasets\\data\\debiased_dataset.csv'
testcsv = 'datasets\\data\\test_dataset.csv'

upscale = True
upscalecoeff = 2

flip = True


def augment(source_path, abspath):
    name = pathlib.Path(source_path).stem.split('_')[0]
    target = pathlib.Path(f"./data/{name}_augmented/")
    path_to_create = abspath / target
    if not os.path.exists(path_to_create):
        os.makedirs(path_to_create)

    n = 0
    df = pd.read_csv(abspath.parent / source_path)
    root_path = abspath.parent
    pthname = name
    print(abspath)
    paths = df.file
    collector = []
    train_text_file = open("data/train_debiased_augmented.txt", "w")
    for (
            image,
            xcent,
            ycent,
            w,
            h,
            name,
            maskon,
            race,
            sex,
            skin_color,
            person_num,
            xmin,
            ymin,
            xmax,
            ymax,
    ) in tqdm.tqdm(
            zip(
                paths,
                df.xcenter,
                df.ycenter,
                df.width,
                df.height,
                df.Img,
                df.Mask_on,
                df.Race,
                df.Sex,
                df.SkinColor,
                df.Person_num,
                df.xmin,
                df.ymin,
                df.xmax,
                df.ymax,
            ),
            total=len(df),
    ):
        im = Image.open(root_path / image)
        width, height = im.size
        n += 1
        while (upscale and width < 1000 and height < 1000):  #makes non-percentage bounding boxes unreliable
            im = im.resize((width * upscalecoeff, height * upscalecoeff))
            width, height = im.size

        if flip:  #makes non-percentage bounding boxes unreliable
            im2 = im.transpose(Image.FLIP_LEFT_RIGHT)
            im2.save(abspath / (f"./data/{pthname}_augmented/flipped_" + name))
            filename = os.path.splitext(abspath / (f"data/{pthname}_augmented/flipped_" + name))[0] + ".txt"
            f = open(filename, "w")
            f.write(str(int(maskon)) + " " + str(abs(1 - xcent)) + " " + str(ycent) + " " + str(w) + " " + str(h))
            f.close()
            collector.append(
                (name, maskon, xcent, ycent, w, h, race, sex, skin_color, person_num, xmin, ymin, xmax, ymax))
            fntmp = (Path(f'data/{pthname}_augmented') / ("flipped_" + name)).as_posix()
            train_text_file.write(f"{fntmp}\n")

        im.save(abspath / (f"data/{pthname}_augmented/" + name))
        filename = os.path.splitext(abspath / (f"data/{pthname}_augmented/" + name))[0] + ".txt"
        f = open(filename, "w")
        f.write(str(int(maskon)) + " " + str(xcent) + " " + str(ycent) + " " + str(w) + " " + str(h))
        f.close()
        collector.append((name, maskon, xcent, ycent, w, h, race, sex, skin_color, person_num, xmin, ymin, xmax, ymax))
        fntmp = (Path(f'data/{pthname}_augmented') / name).as_posix()
        train_text_file.write(f"{fntmp}\n")
    print(n)

    df = pd.DataFrame(
        collector,
        columns="Img, Mask_on, xcenter, ycenter, width, height, Race, Sex, SkinColor, Person_num, xmin, ymin, xmax, ymax"
        .split(", ")).set_index("Img")
    df.to_csv(abspath / f'data/{pthname}_augmented.csv')
    with io.open(abspath / f'object_{pthname}_augmented.data', 'w') as object_data_file:
        content = {
            "classes": 2,
            "train": (f'data/train_{pthname}_augmented.txt'),
            "valid": (f'data/train_test.txt'),
            "names": (f"data/obj.names"),
            "backup": f"backup_{pthname}_augmented/",
        }
        for key, val in content.items():
            object_data_file.write(f"{key} = {val}\n")


# '''

#
augment(debiasedcsv, abspath)
# df = pd.read_csv(abspath / biasedcsv)
# augment(df)
# df = pd.read_csv(abspath / testcsv)
# augment(df)
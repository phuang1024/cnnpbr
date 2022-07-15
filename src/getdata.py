import requests
from subprocess import run, DEVNULL

import cv2
import numpy as np

URL = "https://ambientcg.com/api/v2/full_json"


def query(count, offset, category):
    headers = {
        "User-Agent": "asdf",
    }
    params = {
        "sort": "latest",
        "method": "PBRApproximated,PBRPhotogrammetry,PBRProcedural,PBRMultiAngle",
        "type": "Material",
        "include": "downloadData",
        "limit": count,
        "offset": offset,
        "q": category,
    }
    r = requests.get(URL, headers=headers, params=params)
    r.raise_for_status()

    return r.json()["foundAssets"]


def process_img(args, img):
    size = min(img.shape[0], img.shape[1])
    img = img[:size, :size, :]
    img = cv2.resize(img, (args.size, args.size))
    return img


def get_texture(args, r, data_path, index, total):
    name = r["assetId"]
    print(f"Processing {name} ({index}/{total})")

    download_info = None
    for info in r["downloadFolders"]["default"]["downloadFiletypeCategories"]["zip"]["downloads"]:
        idname = info["attribute"]
        if idname == "2K-PNG":
            download_info = info
            break
    else:
        print(f"  Abort: No 2K PNG")
        return

    url = download_info["fullDownloadPath"]
    zip_path = args.tmp_path / f"{name}.zip"
    print(f"  Downloading {url}")
    proc = run(["wget", url, "-O", zip_path], stdout=DEVNULL, stderr=DEVNULL)
    assert proc.returncode == 0

    unzip_path = args.tmp_path / f"{name}_unzipped"
    unzip_path.mkdir(parents=True, exist_ok=True)
    print(f"  Unzipping {zip_path}")
    proc = run(["unzip", "-o", zip_path, "-d", unzip_path], stdout=DEVNULL, stderr=DEVNULL)
    assert proc.returncode == 0

    maps = {}
    for f in unzip_path.iterdir():
        if f.suffix != ".png":
            continue

        save_name = None
        if "Color" in f.name:
            save_name = "color"
        elif "AmbientOcclusion" in f.name:
            save_name = "ao"
        elif "Displacement" in f.name:
            save_name = "disp"
        elif "Roughness" in f.name:
            save_name = "rough"
        elif "NormalGL" in f.name:
            save_name = "normal"
        if save_name is None:
            continue

        print(f"  Processing map {save_name}")
        img = cv2.imread(str(f))
        img = process_img(args, img)
        maps[save_name] = img

    for m in ("color", "ao", "disp", "rough", "normal"):
        if m not in maps:
            print(f"  Abort: No map {name}")
            return

    data_path = data_path / name
    data_path.mkdir(parents=True, exist_ok=True)
    for m, img in maps.items():
        cv2.imwrite(str(data_path / f"{m}.png"), img)

    print(f"  Done: {name}")


def getdata(args):
    print("Downloading data from AmbientCG")

    results = []

    # Make requests of up to 100 to get all results.
    start_i = 0
    while True:
        stop_i = min(start_i + 100, args.count)
        r = query(stop_i-start_i-1, start_i, args.category)
        if len(r) == 0:
            break

        results.extend(r)
        start_i = stop_i
        if start_i >= args.count:
            break

    path = args.data_path / args.category
    path.mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(results):
        get_texture(args, r, path, i, len(results))

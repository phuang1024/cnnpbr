import argparse
import json
import os
import random
import shutil
import time
from subprocess import Popen, STDOUT, PIPE
from threading import Thread

import requests
from tqdm import tqdm

PARENT = os.path.dirname(os.path.realpath(__file__))

CHUNK_SIZE = 20
DOMAIN = "https://ambientcg.com"
REQUIRED_MAPS = ["color", "disp", "rough", "normal"]


def run(args, cwd="."):
    cwd = os.path.abspath(cwd)
    p = Popen(args, cwd=cwd, stdout=PIPE, stderr=STDOUT)
    p.wait()
    if p.returncode != 0:
        print(p.stdout.read().decode())


def get_asset(output, idname):
    url = f"{DOMAIN}/get?file={idname}_2K-JPG.zip"

    final_dir = os.path.join(output, idname)
    if os.path.isdir(final_dir):
        print(f"{idname} already exists, not downloading.")
        return

    tmp = os.path.join(output, f"tmp{random.randint(0, 1e9)}")
    zip_path = os.path.join(tmp, f"{idname}.zip")
    os.makedirs(tmp, exist_ok=True)

    run(["wget", url, "-O", zip_path])
    run(["unzip", "-o", f"{idname}.zip"], cwd=tmp)

    files = [f for f in os.listdir(tmp) if f.endswith(".jpg")]
    maps = {}
    for f in files:
        name = f.lower()
        if "color" in name:
            maps["color"] = f
        if "displacement" in name:
            maps["disp"] = f
        if "roughness" in name:
            maps["rough"] = f
        if "normalgl" in name:
            maps["normal"] = f

    if len(maps) != len(REQUIRED_MAPS):
        print(f"Failed to process {idname}.")
    else:
        os.makedirs(final_dir)
        for k, v in maps.items():
            os.rename(os.path.join(tmp, v), os.path.join(final_dir, f"{k}.jpg"))

    shutil.rmtree(tmp)

def download(args):
    url = f"{DOMAIN}/api/v2/full_json?sort=latest&type=Material&" + \
        f"limit={args.count}&offset={args.offset}"
    r = requests.get(url, headers={"User-Agent": "asdf"}).json()

    assets = r["foundAssets"]

    threads = [None] * args.jobs
    for asset in (pbar := tqdm(assets)):
        while True:
            time.sleep(0.01)
            for i in range(args.jobs):
                if threads[i] is None or not threads[i].is_alive():
                    break
            else:
                continue
            break

        name = asset["assetId"]
        pbar.set_description(name)

        thread = Thread(target=get_asset, args=(args.output, name))
        thread.start()
        threads[i] = thread


def main():
    parser = argparse.ArgumentParser(description="Physically based rendering texture maps using CNNs.")
    parser.add_argument("-c", "--count", type=int, default=10, help="Number of textures to download.")
    parser.add_argument("-o", "--offset", type=int, default=0, help="Query offset.")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="Number of threads to use.")
    parser.add_argument("--output", required=True, help="Output path.")
    args = parser.parse_args()

    print(f"Downloading {args.count} textures from AmbientCG")
    print("Thanks to AmbientCG for providing 3D assets: https://ambientcg.com")
    print("Textures are copyright AmbientCG, licensed as Creative Commons 0.")
    print("Saving textures to ./data")

    download(args)


if __name__ == "__main__":
    main()

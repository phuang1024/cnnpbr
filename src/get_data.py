#
#  CNNPBR
#  Physically based rendering texture maps using CNNs.
#  Copyright  Patrick Huang  2022
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import argparse
import json
import os
import shutil
from subprocess import Popen, DEVNULL

import requests
from tqdm import trange

PARENT = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(PARENT, "data")

CHUNK_SIZE = 20
DOMAIN = "https://ambientcg.com"


def run(args, cwd="."):
    cwd = os.path.abspath(cwd)
    Popen(args, cwd=cwd, stdout=DEVNULL, stderr=DEVNULL).wait()


def get_asset(idname):
    url = f"{DOMAIN}/get?file={idname}_1K-JPG.zip"

    tmp = os.path.join(DATA, "tmp")
    zip_path = os.path.join(tmp, f"{idname}.zip")
    os.makedirs(tmp, exist_ok=True)

    run(["wget", url, "-O", zip_path])
    run(["unzip", "-o", zip_path], cwd=tmp)

    files = [f for f in os.listdir(tmp) if f.endswith(".jpg")]
    color = None
    disp = None
    for f in files:
        if "color" in f.lower():
            color = f
        if "disp" in f.lower():
            disp = f

    if color is None or f is None:
        print(f"Failed to process {idname}.")
    else:
        final_dir = os.path.join(DATA, idname)
        os.makedirs(final_dir, exist_ok=True)
        os.rename(os.path.join(tmp, color), os.path.join(final_dir, "color.jpg"))
        os.rename(os.path.join(tmp, disp), os.path.join(final_dir, "disp.jpg"))

    shutil.rmtree(tmp)

def download(count):
    # For some reason 403 error
    #url = f"{DOMAIN}/api/v2/full_json?sort=latest&type=PhotoTexturePBR&limit={count}"
    #r = requests.get(url)

    with open("assets.json", "r") as fp:
        r = json.load(fp)

    assets = r["foundAssets"]
    for i in trange(len(assets)):
        idname = assets[i]["assetId"]
        get_asset(idname)


def main():
    parser = argparse.ArgumentParser(description="Physically based rendering texture maps using CNNs.")
    parser.add_argument("-c", "--count", type=int, default=10, help="Number of textures to download.")
    args = parser.parse_args()
    count = args.count

    print(f"Downloading {count} textures from AmbientCG.com")
    print("Thanks to Lennart Demes, creator of AmbientCG, for providing CC0 assets.")
    print("Saving textures to ./data")
    print("Textures are copyright AmbientCG, licensed as Creative Commons 0 1.0 Universal.")

    download(count)


if __name__ == "__main__":
    main()

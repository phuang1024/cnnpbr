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

    final_dir = os.path.join(DATA, idname)
    if os.path.isdir(final_dir):
        print(f"{idname} already exists, not downloading.")
        return
    os.makedirs(final_dir)

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
        os.rename(os.path.join(tmp, color), os.path.join(final_dir, "color.jpg"))
        os.rename(os.path.join(tmp, disp), os.path.join(final_dir, "disp.jpg"))

    shutil.rmtree(tmp)

def download(args):
    url = f"{DOMAIN}/api/v2/full_json?sort=latest&type=Material&limit={args.count}&offset={args.offset}"
    r = requests.get(url, headers={"User-Agent": "asdf"}).json()

    assets = r["foundAssets"]
    for i in (pbar := trange(len(assets))):
        name = assets[i]["assetId"]
        pbar.set_description(name)
        get_asset(name)


def main():
    parser = argparse.ArgumentParser(description="Physically based rendering texture maps using CNNs.")
    parser.add_argument("-c", "--count", type=int, default=10, help="Number of textures to download.")
    parser.add_argument("-o", "--offset", type=int, default=0, help="Query offset.")
    args = parser.parse_args()

    print(f"Downloading {args.count} textures from AmbientCG")
    print("Thanks to AmbientCG for providing 3D assets: https://ambientcg.com")
    print("Textures are copyright AmbientCG, licensed as Creative Commons 0.")
    print("Saving textures to ./data")

    download(args)


if __name__ == "__main__":
    main()

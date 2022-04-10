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
import os

import requests
from tqdm import trange

PARENT = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(PARENT, "data")

CHUNK_SIZE = 10
ENDPOINT = "https://ambientcg.com/api/v2/full_json"


def main():
    parser = argparse.ArgumentParser(description="Physically based rendering texture maps using CNNs.")
    parser.add_argument("-c", "--count", type=int, default=10, help="Number of textures to download.")
    args = parser.parse_args()
    count = args.count

    print(f"Downloading {count} textures from AmbientCG.com")
    print("Thanks to Lennart Demes, creator of AmbientCG, for providing CC0 assets.")
    print("Saving textures to ./data")
    print("Textures are copyright AmbientCG, licensed as Creative Commons 0 1.0 Universal.")


if __name__ == "__main__":
    main()

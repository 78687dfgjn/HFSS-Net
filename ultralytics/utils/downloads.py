import contextlib
import re
import shutil
import subprocess
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from urllib import parse, request

import requests
import torch

from ultralytics.utils import LOGGER, TQDM, checks, clean_url, emojis, is_online, url2file


GITHUB_ASSETS_REPO = "ultralytics/assets"
GITHUB_ASSETS_NAMES = (
    [f"yolov8{k}{suffix}.pt" for k in "nsmlx" for suffix in ("", "-cls", "-seg", "-pose", "-obb")]
    + [f"yolov5{k}{resolution}u.pt" for k in "nsmlx" for resolution in ("", "6")]
    + [f"yolov3{k}u.pt" for k in ("", "-spp", "-tiny")]
    + [f"yolov8{k}-world.pt" for k in "smlx"]
    + [f"yolov8{k}-worldv2.pt" for k in "smlx"]
    + [f"yolov9{k}.pt" for k in "ce"]
    + [f"yolo_nas_{k}.pt" for k in "sml"]
    + [f"sam_{k}.pt" for k in "bl"]
    + [f"FastSAM-{k}.pt" for k in "sx"]
    + [f"rtdetr-{k}.pt" for k in "lx"]
    + ["mobile_sam.pt"]
    + ["calibration_image_sample_data_20x128x128x3_float32.npy.zip"]
)
GITHUB_ASSETS_STEMS = [Path(k).stem for k in GITHUB_ASSETS_NAMES]


def is_url(url, check=False):

















    with contextlib.suppress(Exception):
        url = str(url)
        result = parse.urlparse(url)
        assert all([result.scheme, result.netloc])
        if check:
            with request.urlopen(url) as response:
                return response.getcode() == 200
        return True
    return False


def delete_dsstore(path, files_to_delete=(".DS_Store", "__MACOSX")):


















    for file in files_to_delete:
        matches = list(Path(path).rglob(file))
        LOGGER.info(f"Deleting {file} files: {matches}")
        for f in matches:
            f.unlink()


def zip_directory(directory, compress=True, exclude=(".DS_Store", "__MACOSX"), progress=True):




















    from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

    delete_dsstore(directory)
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")


    files_to_zip = [f for f in directory.rglob("*") if f.is_file() and all(x not in f.name for x in exclude)]
    zip_file = directory.with_suffix(".zip")
    compression = ZIP_DEFLATED if compress else ZIP_STORED
    with ZipFile(zip_file, "w", compression) as f:
        for file in TQDM(files_to_zip, desc=f"Zipping {directory} to {zip_file}...", unit="file", disable=not progress):
            f.write(file, file.relative_to(directory))

    return zip_file


def unzip_file(file, path=None, exclude=(".DS_Store", "__MACOSX"), exist_ok=False, progress=True):



























    from zipfile import BadZipFile, ZipFile, is_zipfile

    if not (Path(file).exists() and is_zipfile(file)):
        raise BadZipFile(f"File '{file}' does not exist or is a bad zip file.")
    if path is None:
        path = Path(file).parent


    with ZipFile(file) as zipObj:
        files = [f for f in zipObj.namelist() if all(x not in f for x in exclude)]
        top_level_dirs = {Path(f).parts[0] for f in files}

        if len(top_level_dirs) > 1 or (len(files) > 1 and not files[0].endswith("/")):

            path = extract_path = Path(path) / Path(file).stem
        else:

            extract_path = path
            path = Path(path) / list(top_level_dirs)[0]


        if path.exists() and any(path.iterdir()) and not exist_ok:

            LOGGER.warning(f"WARNING ⚠️ Skipping {file} unzip as destination directory {path} is not empty.")
            return path

        for f in TQDM(files, desc=f"Unzipping {file} to {Path(path).resolve()}...", unit="file", disable=not progress):

            if ".." in Path(f).parts:
                LOGGER.warning(f"Potentially insecure file path: {f}, skipping extraction.")
                continue
            zipObj.extract(f, extract_path)

    return path


def check_disk_space(url="https://ultralytics.com/assets/coco128.zip", path=Path.cwd(), sf=1.5, hard=True):












    try:
        r = requests.head(url)
        assert r.status_code < 400, f"URL error for {url}: {r.status_code} {r.reason}"
    except Exception:
        return True


    gib = 1 << 30
    data = int(r.headers.get("Content-Length", 0)) / gib
    total, used, free = (x / gib for x in shutil.disk_usage(path))

    if data * sf < free:
        return True


    text = (
        f"WARNING ⚠️ Insufficient free disk space {free:.1f} GB < {data * sf:.3f} GB required, "
        f"Please free {data * sf - free:.1f} GB additional disk space and try again."
    )
    if hard:
        raise MemoryError(text)
    LOGGER.warning(text)
    return False


def get_google_drive_file_info(link):


















    file_id = link.split("/d/")[1].split("/view")[0]
    drive_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    filename = None


    with requests.Session() as session:
        response = session.get(drive_url, stream=True)
        if "quota exceeded" in str(response.content.lower()):
            raise ConnectionError(
                emojis(
                    f"❌  Google Drive file download quota exceeded. "
                    f"Please try again later or download this file manually at {link}."
                )
            )
        for k, v in response.cookies.items():
            if k.startswith("download_warning"):
                drive_url += f"&confirm={v}"
        cd = response.headers.get("content-disposition")
        if cd:
            filename = re.findall('filename="(.+)"', cd)[0]
    return drive_url, filename


def safe_download(
    url,
    file=None,
    dir=None,
    unzip=True,
    delete=False,
    curl=False,
    retry=3,
    min_bytes=1e0,
    exist_ok=False,
    progress=True,
):


























    gdrive = url.startswith("https://drive.google.com/")
    if gdrive:
        url, file = get_google_drive_file_info(url)

    f = Path(dir or ".") / (file or url2file(url))
    if "://" not in str(url) and Path(url).is_file():
        f = Path(url)
    elif not f.is_file():
        desc = f"Downloading {url if gdrive else clean_url(url)} to '{f}'"
        LOGGER.info(f"{desc}...")
        f.parent.mkdir(parents=True, exist_ok=True)
        check_disk_space(url, path=f.parent)
        for i in range(retry + 1):
            try:
                if curl or i > 0:
                    s = "sS" * (not progress)
                    r = subprocess.run(["curl", "-#", f"-{s}L", url, "-o", f, "--retry", "3", "-C", "-"]).returncode
                    assert r == 0, f"Curl return value {r}"
                else:
                    method = "torch"
                    if method == "torch":
                        torch.hub.download_url_to_file(url, f, progress=progress)
                    else:
                        with request.urlopen(url) as response, TQDM(
                            total=int(response.getheader("Content-Length", 0)),
                            desc=desc,
                            disable=not progress,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                        ) as pbar:
                            with open(f, "wb") as f_opened:
                                for data in response:
                                    f_opened.write(data)
                                    pbar.update(len(data))

                if f.exists():
                    if f.stat().st_size > min_bytes:
                        break
                    f.unlink()
            except Exception as e:
                if i == 0 and not is_online():
                    raise ConnectionError(emojis(f"❌  Download failure for {url}. Environment is not online.")) from e
                elif i >= retry:
                    raise ConnectionError(emojis(f"❌  Download failure for {url}. Retry limit reached.")) from e
                LOGGER.warning(f"⚠️ Download failure, retrying {i + 1}/{retry} {url}...")

    if unzip and f.exists() and f.suffix in {"", ".zip", ".tar", ".gz"}:
        from zipfile import is_zipfile

        unzip_dir = (dir or f.parent).resolve()
        if is_zipfile(f):
            unzip_dir = unzip_file(file=f, path=unzip_dir, exist_ok=exist_ok, progress=progress)
        elif f.suffix in {".tar", ".gz"}:
            LOGGER.info(f"Unzipping {f} to {unzip_dir}...")
            subprocess.run(["tar", "xf" if f.suffix == ".tar" else "xfz", f, "--directory", unzip_dir], check=True)
        if delete:
            f.unlink()
        return unzip_dir


def get_github_assets(repo="ultralytics/assets", version="latest", retry=False):


















    if version != "latest":
        version = f"tags/{version}"
    url = f"https://api.github.com/repos/{repo}/releases/{version}"
    r = requests.get(url)
    if r.status_code != 200 and r.reason != "rate limit exceeded" and retry:
        r = requests.get(url)
    if r.status_code != 200:
        LOGGER.warning(f"⚠️ GitHub assets check failure for {url}: {r.status_code} {r.reason}")
        return "", []
    data = r.json()
    return data["tag_name"], [x["name"] for x in data["assets"]]


def attempt_download_asset(file, repo="ultralytics/assets", release="v8.1.0", **kwargs):


















    from ultralytics.utils import SETTINGS


    file = str(file)
    file = checks.check_yolov5u_filename(file)
    file = Path(file.strip().replace("'", ""))
    if file.exists():
        return str(file)
    elif (SETTINGS["weights_dir"] / file).exists():
        return str(SETTINGS["weights_dir"] / file)
    else:

        name = Path(parse.unquote(str(file))).name
        download_url = f"https://github.com/{repo}/releases/download"
        if str(file).startswith(("http:/", "https:/")):
            url = str(file).replace(":/", "://")
            file = url2file(name)
            if Path(file).is_file():
                LOGGER.info(f"Found {clean_url(url)} locally at {file}")
            else:
                safe_download(url=url, file=file, min_bytes=1e5, **kwargs)

        elif repo == GITHUB_ASSETS_REPO and name in GITHUB_ASSETS_NAMES:
            safe_download(url=f"{download_url}/{release}/{name}", file=file, min_bytes=1e5, **kwargs)

        else:
            tag, assets = get_github_assets(repo, release)
            if not assets:
                tag, assets = get_github_assets(repo)
            if name in assets:
                safe_download(url=f"{download_url}/{tag}/{name}", file=file, min_bytes=1e5, **kwargs)

        return str(file)


def download(url, dir=Path.cwd(), unzip=True, delete=False, curl=False, threads=1, retry=3, exist_ok=False):



















    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    if threads > 1:
        with ThreadPool(threads) as pool:
            pool.map(
                lambda x: safe_download(
                    url=x[0],
                    dir=x[1],
                    unzip=unzip,
                    delete=delete,
                    curl=curl,
                    retry=retry,
                    exist_ok=exist_ok,
                    progress=threads <= 1,
                ),
                zip(url, repeat(dir)),
            )
            pool.close()
            pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            safe_download(url=u, dir=dir, unzip=unzip, delete=delete, curl=curl, retry=retry, exist_ok=exist_ok)

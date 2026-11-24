"""Download the pinned optional narration assets into the Wildllama voice store."""
import argparse
from pathlib import Path
from urllib.request import urlopen

ASR = "Qwen3-ASR-0.6B"
INDEX = "IndexTTS-2"
INDEX_FILES = [
    ("IndexTeam/IndexTTS-2", "740dcaff396282ffb241903d150ac011cd4b1ede", "",
     ["config.yaml", "bpe.model", "feat1.pt", "feat2.pt", "gpt.pth", "s2mel.pth", "wav2vec2bert_stats.pt", "LICENSE.txt"]),
    ("facebook/w2v-bert-2.0", "da985ba0987f70aaeb84a80f2851cfac8c697a7b", "hf_cache/w2v-bert-2.0",
     ["config.json", "preprocessor_config.json", "model.safetensors"]),
    ("amphion/MaskGCT", "265c6cef07625665d0c28d2faafb1415562379dc", "hf_cache", ["semantic_codec/model.safetensors"]),
    ("funasr/campplus", "e4b6ede7ce16997aff4ae69fbca1f0175e2afede", "hf_cache", ["campplus_cn_common.bin"]),
    ("nvidia/bigvgan_v2_22khz_80band_256x", "633ff708ed5b74903e86ff1298cf4a98e921c513", "hf_cache/bigvgan",
     ["config.json", "bigvgan_generator.pt"]),
]
def install(root, index=False):
    from huggingface_hub import hf_hub_download, snapshot_download
    snapshot_download("Qwen/"+ASR, revision="5eb144179a02acc5e5ba31e748d22b0cf3e303b0", local_dir=root/ASR,
                      allow_patterns=["*.json","*.txt","*.safetensors"])
    for emotion in ("happy","sad","angry"):
        name=f"tau_{emotion}_avg4spk.pt"
        dest=root/"emotion-vectors"/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        with urlopen("https://raw.githubusercontent.com/danielbrito91/xvector-emotion-arithmetic/342ad7208d7ea151615caa004c9b3e9891d4a241/data/tau/"+name,timeout=30) as response:
            tmp=dest.with_suffix(".part")
            tmp.write_bytes(response.read())
            tmp.replace(dest)
    if index:
        for repo, revision, folder, files in INDEX_FILES:
            for name in files:
                p=Path(hf_hub_download(repo, name, revision=revision, local_dir=root/INDEX/folder))
                if name=="semantic_codec/model.safetensors":
                    p.replace(root/INDEX/folder/"semantic_codec_model.safetensors")
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    parser.add_argument("--index",action="store_true")
    opts=parser.parse_args()
    install(opts.root,opts.index)

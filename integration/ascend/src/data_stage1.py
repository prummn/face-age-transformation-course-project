"""Stage 1: caption-based candidates from Megalith-10M InternVL2 captions (captions used ONLY to select; CC-BY-SA text not used for training).
Keeps only tier A (Asian cue in caption) as in the original run."""
import re, glob, pyarrow.parquet as pq, pyarrow as pa
from multiprocessing import Pool
B = "/data/home/1120231322/scow/face-age"
PERSON = re.compile(r"\b(woman|women|man|men|lady|gentleman|girl|boy|person|people|couple|bride|groom|monk|nun|student|worker|farmer|vendor|elderly|portrait|face|selfie|headshot)\b", re.I)
MINOR = re.compile(r"\b(child|children|kid|kids|baby|babies|toddler|infant|schoolgirl|schoolboy|little (girl|boy)|young (girl|boy)|teen|teenager|newborn|preschool)\b", re.I)
NOTPHOTO = re.compile(r"\b(painting|painted|drawing|illustration|sketch|engraving|etching|lithograph|print of|sculpture|statue|bust|figurine|doll|mannequin|cartoon|poster|stamp|coin|banknote|postcard|manuscript|woodblock|ukiyo|black and white|black-and-white|monochrome|sepia|vintage|historic|antique|old photograph|daguerreotype|tintype|mural|mosaic|relief|carving)\b", re.I)
CROWD = re.compile(r"\b(crowd|audience|parade|team|group of|many people|several people|large group|spectators)\b", re.I)
ASIAN = re.compile(r"\b(asian|east asian|southeast asian|chinese|japanese|korean|vietnamese|thai|filipino|filipina|indonesian|malaysian|singaporean|taiwanese|mongolian|cambodian|burmese|myanmar|lao|laotian|hmong|kimono|yukata|hanbok|qipao|cheongsam|hanfu|ao dai|tokyo|kyoto|osaka|beijing|shanghai|hong kong|taipei|seoul|busan|hanoi|saigon|ho chi minh|bangkok|manila|jakarta|bali|kuala lumpur|chinatown|geisha)\b", re.I)
def do(f):
    t = pq.read_table(f, columns=["url_highres", "caption_internlm2_short", "caption_florence2_short", "caption_sharecap_short", "width", "height"]).to_pylist(); out = []
    for r in t:
        caps = " | ".join(c for c in (r["caption_internlm2_short"], r["caption_florence2_short"], r["caption_sharecap_short"]) if c)
        if min(r["width"] or 0, r["height"] or 0) < 500 or not PERSON.search(caps) or MINOR.search(caps) or NOTPHOTO.search(caps) or CROWD.search(caps) or not ASIAN.search(caps): continue
        out.append(dict(url=r["url_highres"], license="flickr:cc0/pdm/nkcr"))
    return out
if __name__ == "__main__":
    rows = []
    with Pool(48) as p:
        for r in p.imap_unordered(do, sorted(glob.glob(f"{B}/data/megalith_caps/train/*.parquet"))): rows += r
    for i, r in enumerate(rows): r["key"] = "%08d" % i
    pq.write_table(pa.Table.from_pylist(rows), f"{B}/data/stage1_tierA.parquet"); print("STAGE1_DONE", len(rows))

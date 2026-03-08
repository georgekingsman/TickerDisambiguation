"""
build_dataset.py  –  Generate the v1 dataset for Ticker Disambiguation
                     生成股票代码消歧 v1 数据集

Outputs / 输出文件:
    data/train.jsonl          – Training set   | 训练集
    data/val.jsonl            – Validation set | 验证集
    data/test.jsonl           – Test set       | 测试集
    data/ambiguous_eval.jsonl – Ambiguity-focused eval | 歧义专项评估集
"""
import json, random, pathlib

# Fixed seed for reproducibility | 固定随机种子以保证可复现性
random.seed(42)

# Instruction shared by all samples | 所有样本共用的指令
INSTRUCTION = "Resolve the stock ticker symbol from the user request. Return only the ticker."

# ── All samples | 全部样本 ─────────────────────────────────────────────────
samples = []

def add(inp, out):
    """Append one (input, output) sample. | 添加一条（输入, 输出）样本。"""
    samples.append({"instruction": INSTRUCTION, "input": inp, "output": out})

# ═══════════════════════════════════════════════════════
# A. Direct / straightforward mapping  (~40 samples)
#    A. 直接映射样本（约 40 条）
# ═══════════════════════════════════════════════════════

# AAPL
add("Analyze Apple's recent earnings.", "AAPL")
add("Give me a summary of AAPL.", "AAPL")
add("How has Apple stock performed this quarter?", "AAPL")
add("Research Apple Inc for me.", "AAPL")
add("Pull up the latest Apple financials.", "AAPL")
add("What's the outlook for Apple?", "AAPL")

# MSFT
add("What's happening with Microsoft?", "MSFT")
add("Research MSFT for me.", "MSFT")
add("Give me a memo on Microsoft stock.", "MSFT")
add("How is Microsoft doing lately?", "MSFT")

# AMZN
add("Analyze Amazon's latest quarter.", "AMZN")
add("Pull up AMZN data.", "AMZN")
add("How has Amazon been performing?", "AMZN")
add("Give me a report on Amazon.", "AMZN")

# TSLA
add("Research Tesla for me.", "TSLA")
add("How is TSLA doing?", "TSLA")
add("Pull up Tesla stock info.", "TSLA")
add("Give me a memo on Tesla.", "TSLA")
add("What's the latest on Tesla Motors?", "TSLA")

# NVDA
add("Analyze Nvidia's growth.", "NVDA")
add("Research NVIDIA for me.", "NVDA")
add("How has NVDA performed recently?", "NVDA")
add("Give me Nvidia's earnings summary.", "NVDA")

# NFLX
add("What's happening with Netflix?", "NFLX")
add("Research NFLX.", "NFLX")
add("How is Netflix stock doing?", "NFLX")

# JPM
add("Research JPMorgan for me.", "JPM")
add("How is JP Morgan doing?", "JPM")
add("Give me a memo on Chase bank stock.", "JPM")
add("Pull up JPM financials.", "JPM")

# BAC
add("Research Bank of America.", "BAC")
add("How has BofA been performing?", "BAC")
add("Give me BAC data.", "BAC")

# AMD
add("Analyze AMD's recent performance.", "AMD")
add("Research Advanced Micro Devices.", "AMD")
add("How is AMD stock doing?", "AMD")

# INTC
add("Research Intel for me.", "INTC")
add("How has Intel been performing lately?", "INTC")
add("Pull up INTC data.", "INTC")

# PLTR
add("Analyze Palantir's growth.", "PLTR")
add("Research PLTR.", "PLTR")
add("How is Palantir doing?", "PLTR")

# DIS
add("Research Disney stock.", "DIS")
add("How has Walt Disney been performing?", "DIS")
add("Give me a memo on DIS.", "DIS")

# TSM
add("Research TSMC for me.", "TSM")
add("How is Taiwan Semiconductor doing?", "TSM")
add("Pull up TSM financials.", "TSM")

# BABA
add("Research Alibaba.", "BABA")
add("How has BABA performed?", "BABA")
add("Give me a memo on Alibaba stock.", "BABA")

# PDD
add("Research Pinduoduo.", "PDD")
add("How is PDD doing?", "PDD")
add("Give me data on Temu's parent company.", "PDD")

# ═══════════════════════════════════════════════════════
# B. Alias / colloquial expressions  (~35 samples)
#    B. 别名 / 口语化表达（约 35 条）
# ═══════════════════════════════════════════════════════

# META / Facebook aliases | META / Facebook 别名
add("How is Facebook doing these days?", "META")
add("Research Facebook stock.", "META")
add("What's happening with FB?", "META")
add("Give me a report on Meta.", "META")
add("Analyze Meta Platforms.", "META")
add("Pull up Facebook's latest earnings.", "META")
add("Research Meta's ad revenue trends.", "META")
add("How has Facebook stock changed since the rebrand?", "META")

# Google / Alphabet defaults → GOOGL | Google / Alphabet 默认映射到 GOOGL
add("Research Google for me.", "GOOGL")
add("How is Google stock doing?", "GOOGL")
add("Give me a summary on Alphabet.", "GOOGL")
add("Pull up Google's earnings.", "GOOGL")
add("Analyze Alphabet Inc.", "GOOGL")
add("What's the outlook for Google?", "GOOGL")
add("How has Alphabet performed this year?", "GOOGL")
add("Give me a memo on Google stock.", "GOOGL")

# Berkshire defaults → BRK-B | Berkshire 默认映射到 BRK-B
add("Research Berkshire Hathaway.", "BRK-B")
add("How is Berkshire doing?", "BRK-B")
add("Give me a report on Berkshire Hathaway stock.", "BRK-B")
add("What's Warren Buffett's company doing?", "BRK-B")
add("Pull up Berkshire's latest filings.", "BRK-B")
add("Analyze Berkshire Hathaway for me.", "BRK-B")

# Other colloquial | 其他口语化表达
add("How's the Mouse House stock?", "DIS")
add("Research the Bezos company.", "AMZN")
add("What's Elon's car company doing?", "TSLA")
add("How is Jensen Huang's company performing?", "NVDA")
add("Give me a memo on Zuckerberg's company.", "META")
add("Research Nadella's company.", "MSFT")

# ═══════════════════════════════════════════════════════
# C. Ambiguity / tricky samples  (~55 samples)
#    C. 歧义 / 陷阱样本（约 55 条）
# ═══════════════════════════════════════════════════════

# --- Google/Alphabet class A vs class C | A类 vs C类 ---
add("Give me a quick memo on Google class C over the last 6 months.", "GOOG")
add("Research Alphabet class A for me.", "GOOGL")
add("How has Google class A performed recently?", "GOOGL")
add("Analyze Google's class C shares.", "GOOG")
add("What's the performance of Alphabet class C?", "GOOG")
add("Pull up Alphabet class A stock data.", "GOOGL")
add("Compare Google class C with the market.", "GOOG")
add("Give me a report on Google's class A shares.", "GOOGL")
add("How is GOOG doing?", "GOOG")
add("Research GOOGL.", "GOOGL")
add("Analyze the non-voting shares of Google.", "GOOG")
add("How are Google's voting shares performing?", "GOOGL")
add("Give me data on Alphabet's class C stock.", "GOOG")
add("Research Google's non-voting class C.", "GOOG")
add("Pull up the class A Alphabet shares.", "GOOGL")

# --- Berkshire class A vs class B | A类 vs B类 ---
add("How is Berkshire B doing recently?", "BRK-B")
add("Research Berkshire class A.", "BRK-A")
add("Analyze Berkshire Hathaway class B shares.", "BRK-B")
add("Give me a report on Berkshire class A stock.", "BRK-A")
add("Pull up Berkshire Hathaway class B data.", "BRK-B")
add("How has BRK-A performed?", "BRK-A")
add("Research BRK-B for me.", "BRK-B")
add("What's the price of Berkshire Hathaway class A?", "BRK-A")
add("Analyze the class B shares of Berkshire.", "BRK-B")
add("How are Berkshire's class A shares doing?", "BRK-A")
add("Give me a memo on Berkshire Hathaway's affordable shares.", "BRK-B")
add("Research Buffett's class A shares.", "BRK-A")

# --- Ticker format normalization | 代码格式标准化 ---
add("How is BRK.B doing?", "BRK-B")
add("Pull up BRK.A data.", "BRK-A")
add("Research BRK/B.", "BRK-B")
add("Analyze BRK.B performance.", "BRK-B")
add("Give me a report on BRK.A.", "BRK-A")
add("What's happening with BRK B?", "BRK-B")
add("How has BRK/A performed?", "BRK-A")

# --- Mixed ambiguity: colloquial + class | 混合歧义：口语化 + 股类 ---
add("Tell me about Google's cheaper share class.", "GOOG")
add("Research the expensive Berkshire shares.", "BRK-A")
add("How is the retail-friendly Berkshire stock?", "BRK-B")
add("Give me info on Google's original IPO share class.", "GOOGL")
add("Analyze the Alphabet shares that don't have voting rights.", "GOOG")
add("Pull up the voting class of Alphabet stock.", "GOOGL")
add("What's the deal with Facebook now that it's called Meta?", "META")
add("Research the company formerly known as Facebook.", "META")
add("How is the old Facebook ticker doing?", "META")

# --- Edge cases: ticker in sentence | 边界情况：句中直接提到代码 ---
add("I'm interested in GOOGL, can you research it?", "GOOGL")
add("Quick analysis of GOOG please.", "GOOG")
add("What about BRK-A? How's it doing?", "BRK-A")
add("Look into BRK-B for my portfolio.", "BRK-B")


# ═══════════════════════════════════════════════════════
# Build ambiguity eval set (separate, hand-picked)
# 构建歧义评估集（独立的、手工挑选的）
# ═══════════════════════════════════════════════════════

ambiguous_eval = [
    {"instruction": INSTRUCTION, "input": "Summarize Google class C for me.", "output": "GOOG"},
    {"instruction": INSTRUCTION, "input": "Research Alphabet's class A shares.", "output": "GOOGL"},
    {"instruction": INSTRUCTION, "input": "How has Google class A done this year?", "output": "GOOGL"},
    {"instruction": INSTRUCTION, "input": "Give me class C Alphabet data.", "output": "GOOG"},
    {"instruction": INSTRUCTION, "input": "Analyze Berkshire Hathaway class A.", "output": "BRK-A"},
    {"instruction": INSTRUCTION, "input": "How is Berkshire's class B stock?", "output": "BRK-B"},
    {"instruction": INSTRUCTION, "input": "Research BRK.B for me.", "output": "BRK-B"},
    {"instruction": INSTRUCTION, "input": "Pull up BRK.A financials.", "output": "BRK-A"},
    {"instruction": INSTRUCTION, "input": "Give me a report on Facebook.", "output": "META"},
    {"instruction": INSTRUCTION, "input": "How is the stock formerly known as FB?", "output": "META"},
    {"instruction": INSTRUCTION, "input": "Research Google's non-voting stock.", "output": "GOOG"},
    {"instruction": INSTRUCTION, "input": "Analyze Alphabet's voting shares.", "output": "GOOGL"},
    {"instruction": INSTRUCTION, "input": "What about BRK/A performance?", "output": "BRK-A"},
    {"instruction": INSTRUCTION, "input": "How are Berkshire class B shares?", "output": "BRK-B"},
    {"instruction": INSTRUCTION, "input": "Give me data on Google.", "output": "GOOGL"},
    {"instruction": INSTRUCTION, "input": "Research Berkshire Hathaway.", "output": "BRK-B"},
    {"instruction": INSTRUCTION, "input": "Analyze Meta Platforms Inc.", "output": "META"},
    {"instruction": INSTRUCTION, "input": "What's happening with Alphabet?", "output": "GOOGL"},
    {"instruction": INSTRUCTION, "input": "Research the cheaper Google share class.", "output": "GOOG"},
    {"instruction": INSTRUCTION, "input": "How is BRK B performing?", "output": "BRK-B"},
]


# ═══════════════════════════════════════════════════════
# Shuffle and split main samples | 打乱并切分主样本集
# Split ratio: train 80 / val 20 / test remainder
# 切分比例：训练 80 / 验证 20 / 测试 剩余
# ═══════════════════════════════════════════════════════

random.shuffle(samples)
n = len(samples)
print(f"Total main samples / 主样本总数: {n}")

train_end = 80
val_end = train_end + 20
train = samples[:train_end]
val = samples[train_end:val_end]
test = samples[val_end:]

print(f"Train / 训练: {len(train)}, Val / 验证: {len(val)}, Test / 测试: {len(test)}")
print(f"Ambiguous eval / 歧义评估: {len(ambiguous_eval)}")

# ── Write JSONL files | 写入 JSONL 文件 ────────────────
data_dir = pathlib.Path("data")

def write_jsonl(path, data):
    """Write a list of dicts to a JSONL file. | 将字典列表写入 JSONL 文件。"""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

write_jsonl(data_dir / "train.jsonl", train)
write_jsonl(data_dir / "val.jsonl", val)
write_jsonl(data_dir / "test.jsonl", test)
write_jsonl(data_dir / "ambiguous_eval.jsonl", ambiguous_eval)

# ── Print label distribution | 打印标签分布 ───────────
from collections import Counter
dist = Counter(s["output"] for s in samples)
print("\nLabel distribution (all main samples) / 标签分布（全部主样本）:")
for sym, cnt in sorted(dist.items(), key=lambda x: -x[1]):
    print(f"  {sym}: {cnt}")

amb_dist = Counter(s["output"] for s in ambiguous_eval)
print("\nAmbiguous eval distribution / 歧义评估集分布:")
for sym, cnt in sorted(amb_dist.items(), key=lambda x: -x[1]):
    print(f"  {sym}: {cnt}")

print("\nDone. Files written to data/ | 完成。文件已写入 data/ 目录")

"""
build_dataset_v2.py  –  Generate v2 training data for LoRA fine-tuning
                        生成用于 LoRA 微调的 v2 训练数据

Changes from v1 / 相比 v1 的改进:
  - Targeted coverage of share-class disambiguation | 股类歧义定向覆盖
  - CEO/founder and colloquial references          | CEO/创始人及口语化引用
  - Indirect product/service descriptions           | 间接产品/服务描述
  - Format noise handling                           | 格式噪声处理
  - Output discipline reinforcement                 | 输出规范性强化
  - Automated leakage check against frozen eval     | 针对冻结评估集的自动泄漏检查

Outputs / 输出文件:
    data/train_v2.jsonl   – Training set   (~420 samples)
    data/val_v2.jsonl     – Validation set (~80 samples)
    data/dev_hard.jsonl   – Dev hard set for tuning (~40 samples)

Usage / 用法:
    python scripts/build_dataset_v2.py
"""

import json
import random
import pathlib
from collections import Counter

# Fixed seed for reproducibility | 固定随机种子
random.seed(42)

INSTRUCTION = "Resolve the stock ticker symbol from the user request. Return only the ticker."

# ── Helpers ────────────────────────────────────────────
samples = []


def add(inp: str, out: str):
    """Append one (input, output) sample."""
    samples.append({"instruction": INSTRUCTION, "input": inp, "output": out})


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, data):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ═══════════════════════════════════════════════════════════════════
# CATEGORY A: Direct / Explicit Requests  (~100 samples)
# A 类：直接显式请求
# Purpose: Baseline coverage of all 20 symbols with standard phrasing
# ═══════════════════════════════════════════════════════════════════

# ── AAPL ──
add("Analyze Apple's recent earnings.", "AAPL")
add("Give me a summary of AAPL.", "AAPL")
add("How has Apple stock performed this quarter?", "AAPL")
add("Research Apple Inc for me.", "AAPL")
add("Pull up the latest Apple financials.", "AAPL")
add("What's the outlook for Apple?", "AAPL")

# ── MSFT ──
add("What's happening with Microsoft?", "MSFT")
add("Research MSFT for me.", "MSFT")
add("Give me a memo on Microsoft stock.", "MSFT")
add("How is Microsoft doing lately?", "MSFT")
add("Analyze Microsoft Corp.", "MSFT")

# ── AMZN ──
add("Analyze Amazon's latest quarter.", "AMZN")
add("Pull up AMZN data.", "AMZN")
add("How has Amazon been performing?", "AMZN")
add("Give me a report on Amazon.", "AMZN")
add("Research Amazon.com stock.", "AMZN")

# ── META ──
add("Give me a report on Meta.", "META")
add("Analyze Meta Platforms.", "META")
add("How is Meta stock doing?", "META")
add("Research Meta Platforms Inc.", "META")
add("Pull up META data.", "META")

# ── TSLA ──
add("Research Tesla for me.", "TSLA")
add("How is TSLA doing?", "TSLA")
add("Pull up Tesla stock info.", "TSLA")
add("Give me a memo on Tesla.", "TSLA")
add("What's the latest on Tesla Motors?", "TSLA")
add("Analyze Tesla Inc.", "TSLA")

# ── NVDA ──
add("Analyze Nvidia's growth.", "NVDA")
add("Research NVIDIA for me.", "NVDA")
add("How has NVDA performed recently?", "NVDA")
add("Give me Nvidia's earnings summary.", "NVDA")
add("Pull up NVIDIA stock.", "NVDA")

# ── NFLX ──
add("What's happening with Netflix?", "NFLX")
add("Research NFLX.", "NFLX")
add("How is Netflix stock doing?", "NFLX")
add("Analyze Netflix Inc.", "NFLX")
add("Give me a memo on Netflix.", "NFLX")

# ── JPM ──
add("Research JPMorgan for me.", "JPM")
add("How is JP Morgan doing?", "JPM")
add("Give me a memo on Chase bank stock.", "JPM")
add("Pull up JPM financials.", "JPM")
add("Analyze JPMorgan Chase.", "JPM")

# ── BAC ──
add("Research Bank of America.", "BAC")
add("How has BofA been performing?", "BAC")
add("Give me BAC data.", "BAC")
add("Analyze Bank of America Corp.", "BAC")
add("Pull up BAC stock info.", "BAC")

# ── AMD ──
add("Analyze AMD's recent performance.", "AMD")
add("Research Advanced Micro Devices.", "AMD")
add("How is AMD stock doing?", "AMD")
add("Give me data on AMD.", "AMD")

# ── INTC ──
add("Research Intel for me.", "INTC")
add("How has Intel been performing lately?", "INTC")
add("Pull up INTC data.", "INTC")
add("Analyze Intel Corporation.", "INTC")

# ── PLTR ──
add("Analyze Palantir's growth.", "PLTR")
add("Research PLTR.", "PLTR")
add("How is Palantir doing?", "PLTR")
add("Give me data on Palantir Technologies.", "PLTR")

# ── DIS ──
add("Research Disney stock.", "DIS")
add("How has Walt Disney been performing?", "DIS")
add("Give me a memo on DIS.", "DIS")
add("Analyze the Walt Disney Company.", "DIS")

# ── TSM ──
add("Research TSMC for me.", "TSM")
add("How is Taiwan Semiconductor doing?", "TSM")
add("Pull up TSM financials.", "TSM")
add("Analyze TSMC stock.", "TSM")

# ── BABA ──
add("Research Alibaba.", "BABA")
add("How has BABA performed?", "BABA")
add("Give me a memo on Alibaba stock.", "BABA")
add("Analyze Alibaba Group.", "BABA")

# ── PDD ──
add("Research Pinduoduo.", "PDD")
add("How is PDD doing?", "PDD")
add("Give me data on Temu's parent company.", "PDD")
add("Analyze PDD Holdings.", "PDD")

# ── GOOGL / GOOG defaults ──
add("Research Google for me.", "GOOGL")
add("How is Google stock doing?", "GOOGL")
add("Give me a summary on Alphabet.", "GOOGL")
add("Pull up Google's earnings.", "GOOGL")
add("Analyze Alphabet Inc.", "GOOGL")
add("What's the outlook for Google?", "GOOGL")
add("How has Alphabet performed this year?", "GOOGL")
add("Give me a memo on Google stock.", "GOOGL")
add("Research Alphabet.", "GOOGL")
add("What's happening with Google?", "GOOGL")

# ── BRK-B defaults ──
add("Research Berkshire Hathaway.", "BRK-B")
add("How is Berkshire doing?", "BRK-B")
add("Give me a report on Berkshire Hathaway stock.", "BRK-B")
add("What's Warren Buffett's company doing?", "BRK-B")
add("Pull up Berkshire's latest filings.", "BRK-B")
add("Analyze Berkshire Hathaway for me.", "BRK-B")
add("Research Berkshire.", "BRK-B")
add("Give me data on Berkshire Hathaway.", "BRK-B")


# ═══════════════════════════════════════════════════════════════════
# CATEGORY B: Share Class Disambiguation  (~120 samples)
# B 类：股类歧义消解
# Purpose: Teach Google class A/C and Berkshire class A/B resolution
# ═══════════════════════════════════════════════════════════════════

# ── Google/Alphabet CLASS A → GOOGL ──
add("Research Alphabet class A for me.", "GOOGL")
add("How has Google class A performed recently?", "GOOGL")
add("Pull up Alphabet class A stock data.", "GOOGL")
add("Give me a report on Google's class A shares.", "GOOGL")
add("How are Google's voting shares performing?", "GOOGL")
add("Pull up the class A Alphabet shares.", "GOOGL")
add("I'm interested in GOOGL, can you research it?", "GOOGL")
add("Research GOOGL.", "GOOGL")
add("Analyze Alphabet's class A equity.", "GOOGL")
add("Give me data on Google's voting class.", "GOOGL")
add("How did Google's A-class shares close?", "GOOGL")
add("Research the voting Alphabet shares.", "GOOGL")
add("Give me Alphabet's class A performance.", "GOOGL")
add("Analyze the Google shares that have voting rights.", "GOOGL")
add("What's the price of Alphabet's class A stock?", "GOOGL")
add("How is Google's voting stock performing?", "GOOGL")
add("Pull up the A-class Alphabet equity.", "GOOGL")
add("Give me a report on Google class A stock.", "GOOGL")
add("Research the voting class of Google shares.", "GOOGL")
add("How are the class A Alphabet shares?", "GOOGL")
add("Look up Alphabet stock with voting rights.", "GOOGL")
add("Analyze the class A shares of Alphabet Inc.", "GOOGL")
add("Give me data on Google's share class with votes.", "GOOGL")
add("Research Google class A.", "GOOGL")
add("Pull up class A Google data.", "GOOGL")

# ── Google/Alphabet CLASS C → GOOG ──
add("Give me a quick memo on Google class C over the last 6 months.", "GOOG")
add("Analyze Google's class C shares.", "GOOG")
add("What's the performance of Alphabet class C?", "GOOG")
add("Compare Google class C with the market.", "GOOG")
add("How is GOOG doing?", "GOOG")
add("Analyze the non-voting shares of Google.", "GOOG")
add("Give me data on Alphabet's class C stock.", "GOOG")
add("Research Google's non-voting class C.", "GOOG")
add("Quick analysis of GOOG please.", "GOOG")
add("Research Alphabet's class C equity.", "GOOG")
add("Give me data on Google's non-voting shares.", "GOOG")
add("How are Google's class C shares performing?", "GOOG")
add("Analyze the Alphabet class C stock.", "GOOG")
add("Pull up non-voting Google data.", "GOOG")
add("I need info on Google's class C.", "GOOG")
add("Look up Alphabet stock without voting rights.", "GOOG")
add("How did Google's C-class shares close?", "GOOG")
add("Research the non-voting Alphabet shares.", "GOOG")
add("Give me Alphabet's class C performance.", "GOOG")
add("Analyze the Google shares without voting rights.", "GOOG")
add("How is Google's non-voting stock performing?", "GOOG")
add("Pull up the C-class Alphabet equity.", "GOOG")
add("Analyze the no-vote Alphabet shares.", "GOOG")
add("Research the non-voting class of Google shares.", "GOOG")
add("How are the class C Alphabet shares?", "GOOG")

# ── Indirect Google class references ──
add("Tell me about Google's cheaper share class.", "GOOG")
add("Give me info on Google's original IPO share class.", "GOOGL")
add("Pull up the voting class of Alphabet stock.", "GOOGL")
add("Research the Google stock class that carries zero voting power.", "GOOG")
add("How is the Alphabet equity with full shareholder voting?", "GOOGL")
add("Analyze the Alphabet stock where shareholders get a vote.", "GOOGL")
add("Pull up the Google shares designed without governance rights.", "GOOG")
add("Research Alphabet's share class that comes with a vote.", "GOOGL")
add("How are the Google shares that have no say in corporate matters?", "GOOG")
add("Give me the Alphabet class with full voting participation.", "GOOGL")
add("Research the Alphabet stock with no governance stake.", "GOOG")
add("Analyze Google's IPO-era share class.", "GOOGL")

# ── Berkshire CLASS A → BRK-A ──
add("Research Berkshire class A.", "BRK-A")
add("Give me a report on Berkshire class A stock.", "BRK-A")
add("How has BRK-A performed?", "BRK-A")
add("What's the price of Berkshire Hathaway class A?", "BRK-A")
add("How are Berkshire's class A shares doing?", "BRK-A")
add("Research Buffett's class A shares.", "BRK-A")
add("What about BRK-A? How's it doing?", "BRK-A")
add("Research Berkshire Hathaway's class A shares.", "BRK-A")
add("Give me data on the high-priced Berkshire stock.", "BRK-A")
add("How is BRK class A doing?", "BRK-A")
add("Analyze Berkshire's premium share class.", "BRK-A")
add("Pull up the full-price Berkshire Hathaway data.", "BRK-A")
add("Research Buffett's original Berkshire shares.", "BRK-A")
add("Give me info on BRK-A stock.", "BRK-A")
add("How are the class A Berkshire shares performing?", "BRK-A")
add("Analyze the institutional-grade Berkshire stock.", "BRK-A")
add("Research the higher-denomination Berkshire class.", "BRK-A")
add("Give me Berkshire Hathaway class A data.", "BRK-A")
add("Pull up Berkshire's most expensive shares.", "BRK-A")
add("Analyze Berkshire Hathaway's full-size shares.", "BRK-A")

# ── Berkshire CLASS B → BRK-B ──
add("How is Berkshire B doing recently?", "BRK-B")
add("Analyze Berkshire Hathaway class B shares.", "BRK-B")
add("Pull up Berkshire Hathaway class B data.", "BRK-B")
add("Research BRK-B for me.", "BRK-B")
add("Analyze the class B shares of Berkshire.", "BRK-B")
add("Give me a memo on Berkshire Hathaway's affordable shares.", "BRK-B")
add("Look into BRK-B for my portfolio.", "BRK-B")
add("Research Berkshire Hathaway's class B shares.", "BRK-B")
add("Give me data on the affordable Berkshire stock.", "BRK-B")
add("How is BRK class B doing?", "BRK-B")
add("Analyze Berkshire's retail share class.", "BRK-B")
add("Pull up the retail-friendly Berkshire data.", "BRK-B")
add("Research Buffett's accessible Berkshire shares.", "BRK-B")
add("Give me info on BRK-B stock.", "BRK-B")
add("How are the class B Berkshire shares performing?", "BRK-B")
add("Analyze the retail-grade Berkshire stock.", "BRK-B")
add("Research the lower-denomination Berkshire class.", "BRK-B")
add("Give me Berkshire Hathaway class B data.", "BRK-B")
add("Pull up Berkshire's more affordable shares.", "BRK-B")
add("Analyze Berkshire Hathaway's split shares.", "BRK-B")

# ── Indirect Berkshire class references ──
add("Research the expensive Berkshire shares.", "BRK-A")
add("How is the retail-friendly Berkshire stock?", "BRK-B")
add("How is the Berkshire stock regular investors can afford?", "BRK-B")
add("Research the Berkshire class that trades for hundreds of thousands.", "BRK-A")
add("Give me data on Buffett's more accessible share class.", "BRK-B")
add("Analyze the Berkshire stock that debuted as a lower entry point.", "BRK-B")
add("Pull up the Berkshire Hathaway class with the massive price tag.", "BRK-A")
add("Research the Berkshire shares created for everyday investors.", "BRK-B")
add("How is the original undivided Berkshire stock?", "BRK-A")
add("Give me info on the Berkshire class that most brokerage clients hold.", "BRK-B")
add("Analyze Berkshire's entry-level share class.", "BRK-B")
add("Research the Berkshire Hathaway stock with a three-digit price.", "BRK-B")

# ── Ticker format normalization (BRK variants) ──
add("How is BRK.B doing?", "BRK-B")
add("Pull up BRK.A data.", "BRK-A")
add("Research BRK/B.", "BRK-B")
add("Analyze BRK.B performance.", "BRK-B")
add("Give me a report on BRK.A.", "BRK-A")
add("What's happening with BRK B?", "BRK-B")
add("How has BRK/A performed?", "BRK-A")
add("Pull up brk.b financials.", "BRK-B")
add("Research Brk.A stock.", "BRK-A")
add("How is BRK/B doing today?", "BRK-B")
add("Analyze BRK.A for my portfolio.", "BRK-A")
add("Give me BRK B data.", "BRK-B")
add("Research BRK/A.", "BRK-A")
add("What's happening with Brk-B?", "BRK-B")


# ═══════════════════════════════════════════════════════════════════
# CATEGORY C: Brand / Alias / Old Name / CEO References  (~100 samples)
# C 类：品牌/别名/旧名称/CEO 引用
# Purpose: Teach alias resolution, brand history, name mapping
# ═══════════════════════════════════════════════════════════════════

# ── META / Facebook aliases ──
add("How is Facebook doing these days?", "META")
add("Research Facebook stock.", "META")
add("What's happening with FB?", "META")
add("Pull up Facebook's latest earnings.", "META")
add("Research Meta's ad revenue trends.", "META")
add("How has Facebook stock changed since the rebrand?", "META")
add("Analyze the company that used to be Facebook.", "META")
add("Pull up FB stock information.", "META")
add("Research the social media company that changed its name.", "META")
add("How is the Facebook parent company?", "META")
add("Give me info on the former Facebook stock.", "META")
add("Analyze the company behind Facebook and Instagram.", "META")
add("Give me the ticker for Facebook.", "META")
add("Research the owner of Instagram.", "META")
add("Pull up the WhatsApp parent company.", "META")
add("How is the company behind Oculus VR doing?", "META")
add("How is Meta doing since the Facebook rebrand?", "META")
add("Give me data on the company running Facebook.", "META")
add("Pull up the rebranded Facebook shares.", "META")
add("What's the deal with Facebook now that it's called Meta?", "META")
add("Research the company formerly known as Facebook.", "META")
add("How is the old Facebook ticker doing?", "META")

# ── CEO / Founder references (different phrasings from hard_eval) ──
add("Research the company Jeff Bezos founded.", "AMZN")
add("How is the firm Tim Cook runs?", "AAPL")
add("Give me data on Andy Jassy's company.", "AMZN")
add("Analyze the stock of Mark Zuckerberg's firm.", "META")
add("Pull up data on Elon Musk's auto company.", "TSLA")
add("Research Warren Buffett's investment firm.", "BRK-B")
add("How is Jensen Huang's semiconductor firm?", "NVDA")
add("Give me info on Satya Nadella's tech company.", "MSFT")
add("Analyze the company Lisa Su leads.", "AMD")
add("Research Jamie Dimon's banking firm.", "JPM")
add("How is Bob Iger's entertainment company?", "DIS")
add("Research the company founded by Bill Gates.", "MSFT")
add("How is the company Reed Hastings co-founded?", "NFLX")
add("Give me data on the firm started by two Stanford PhDs.", "GOOGL")
add("Analyze the company Steve Jobs co-founded.", "AAPL")
add("Research the Zuckerberg-led social media firm.", "META")
add("How is the company headed by Tim Cook performing?", "AAPL")
add("Research the empire Warren Buffett built.", "BRK-B")
add("Give me info on Jack Ma's e-commerce company.", "BABA")
add("Pull up the company that Musk took over.", "TSLA")

# ── Other colloquial / nickname references ──
add("How's the Mouse House stock?", "DIS")
add("Research the Bezos company.", "AMZN")
add("What's Elon's car company doing?", "TSLA")
add("How is Jensen Huang's company performing?", "NVDA")
add("Give me a memo on Zuckerberg's company.", "META")
add("Research Nadella's company.", "MSFT")
add("How is the House of Mouse stock?", "DIS")
add("Research Disney's kingdom stock.", "DIS")
add("Give me data on Mickey's company.", "DIS")
add("How is the tech giant from Redmond?", "MSFT")
add("Analyze the king of electric vehicles.", "TSLA")
add("How is the GPU giant doing?", "NVDA")
add("Research the king of search engines.", "GOOGL")
add("Give me data on the social media behemoth.", "META")
add("How is America's largest bank?", "JPM")
add("Research the Oracle of Omaha's firm.", "BRK-B")
add("Analyze the chip company known for Ryzen processors.", "AMD")
add("How is the electric car pioneer?", "TSLA")
add("Give me data on the everything store.", "AMZN")
add("Research the e-commerce titan from Seattle.", "AMZN")
add("Analyze the streaming pioneer.", "NFLX")
add("How is the AI chip darling?", "NVDA")
add("Research the Chinese e-commerce company behind Alibaba.com.", "BABA")
add("How is the budget shopping app company PDD?", "PDD")

# ── Full company name variations ──
add("Research TSMC stock.", "TSM")
add("How is Taiwan Semi doing?", "TSM")
add("Analyze Walt Disney Company.", "DIS")
add("Research Advanced Micro Devices.", "AMD")
add("How is Bank of America Corp doing?", "BAC")
add("Give me data on JPMorgan Chase.", "JPM")
add("Analyze JP Morgan & Co.", "JPM")
add("Research Tesla Inc.", "TSLA")
add("How is NVIDIA Corp performing?", "NVDA")
add("Research Intel Corporation.", "INTC")
add("Analyze Palantir Technologies.", "PLTR")
add("Give me info on Pinduoduo Holdings.", "PDD")
add("Research the Alibaba Group.", "BABA")


# ═══════════════════════════════════════════════════════════════════
# CATEGORY D: Indirect Descriptions  (~80 samples)
# D 类：间接描述
# Purpose: Teach resolution from product, service, and industry refs
# ═══════════════════════════════════════════════════════════════════

# ── Product / service references ──
add("Give me data on the company behind the iPhone.", "AAPL")
add("Research the iPad maker.", "AAPL")
add("How is the company behind iCloud?", "AAPL")
add("Analyze the creator of the App Store.", "AAPL")
add("Research the company that built Azure cloud.", "MSFT")
add("How is the Windows operating system maker?", "MSFT")
add("Give me info on the company behind Office 365.", "MSFT")
add("Analyze the company behind the Surface laptop.", "MSFT")
add("Research the company that runs the Kindle store.", "AMZN")
add("How is the firm that operates the AWS cloud?", "AMZN")
add("Analyze the company that acquired Whole Foods.", "AMZN")
add("Give me data on the company behind Alexa.", "AMZN")
add("Research the company behind the Fire TV Stick.", "AMZN")
add("Give me data on the company behind the Messenger app.", "META")
add("Analyze the company that runs Instagram.", "META")
add("Research the maker of Model 3 and Model Y.", "TSLA")
add("How is the company behind Autopilot?", "TSLA")
add("Give me info on the company that makes the H100 GPU.", "NVDA")
add("Analyze the creator of CUDA technology.", "NVDA")
add("Research the company behind GeForce GPUs.", "NVDA")
add("How is the company behind Stranger Things?", "NFLX")
add("Give me data on the company operating Chase Bank.", "JPM")
add("Research the bank that absorbed Merrill Lynch.", "BAC")
add("How is the maker of Ryzen and EPYC chips?", "AMD")
add("Give me info on the maker of Core processors.", "INTC")
add("Analyze the firm behind Gotham analytics.", "PLTR")
add("Research the company behind Disney+ streaming.", "DIS")
add("How is the theme park empire stock?", "DIS")
add("Give me data on the world's largest chip foundry.", "TSM")
add("Analyze the company that fabricates chips for Apple and Nvidia.", "TSM")
add("Research the company behind Taobao marketplace.", "BABA")
add("How is the parent of Temu?", "PDD")

# ── Industry / sector descriptions ──
add("How is the biggest online retailer doing?", "AMZN")
add("Research the dominant cloud infrastructure provider.", "AMZN")
add("Analyze the leading EV manufacturer.", "TSLA")
add("Give me data on the top GPU company.", "NVDA")
add("How is the dominant web search company?", "GOOGL")
add("Research the largest streaming video platform.", "NFLX")
add("Analyze the largest US bank.", "JPM")
add("Give me info on the leading social media conglomerate.", "META")
add("How is the top data analytics defense contractor?", "PLTR")
add("Research the biggest theme park operator.", "DIS")
add("Analyze the world's largest contract chip manufacturer.", "TSM")
add("Give me data on the biggest Chinese e-commerce platform.", "BABA")
add("How is the biggest holding company from Omaha?", "BRK-B")
add("Research the top consumer electronics brand globally.", "AAPL")
add("Analyze the enterprise software giant.", "MSFT")
add("Give me info on the leading x86 processor maker.", "INTC")
add("How is the challenger chip designer in CPUs and GPUs?", "AMD")
add("Research the budget e-commerce app from China.", "PDD")
add("Analyze the company that dominates AI accelerator chips.", "NVDA")
add("How is the leading search advertising company?", "GOOGL")

# ── Contextual / knowledge references ──
add("How is the company that owns YouTube?", "GOOGL")
add("Give me data on the company behind Prime delivery.", "AMZN")
add("Research the company that owns ESPN and Pixar.", "DIS")
add("Analyze the conglomerate that owns Geico and See's Candies.", "BRK-B")
add("Research the company known for its campus in Cupertino.", "AAPL")
add("How is the social media firm headquartered in Menlo Park?", "META")
add("Analyze the company behind the Xbox gaming console.", "MSFT")
add("Give me data on the company behind Prime Video.", "AMZN")
add("Research the Omaha-based holding company.", "BRK-B")
add("How is the company behind the Model S?", "TSLA")


# ═══════════════════════════════════════════════════════════════════
# CATEGORY E: Format / Case Normalization  (~50 samples)
# E 类：格式/大小写标准化
# Purpose: Handle ticker variations, case noise, minimal input
# ═══════════════════════════════════════════════════════════════════

# ── Lowercase ticker inputs ──
add("How is aapl doing?", "AAPL")
add("Research msft.", "MSFT")
add("Pull up googl data.", "GOOGL")
add("Analyze goog.", "GOOG")
add("Give me amzn info.", "AMZN")
add("How is tsla performing?", "TSLA")
add("Research nvda.", "NVDA")
add("Pull up meta stock.", "META")
add("Analyze nflx.", "NFLX")
add("Give me jpm data.", "JPM")
add("Research bac.", "BAC")
add("How is amd doing?", "AMD")
add("Pull up intc financials.", "INTC")
add("Analyze pltr.", "PLTR")
add("Research dis.", "DIS")
add("Give me tsm data.", "TSM")
add("How is baba performing?", "BABA")
add("Research pdd.", "PDD")

# ── Mixed-case and noisy inputs ──
add("Pull up Goog.", "GOOG")
add("How is Googl stock?", "GOOGL")
add("What about TSLA??", "TSLA")
add("Give me brk-b???", "BRK-B")
add("NVDA pls", "NVDA")
add("brk.b", "BRK-B")
add("googl", "GOOGL")
add("aapl", "AAPL")
add("How is tsm", "TSM")
add("research amzn", "AMZN")
add("GIVE ME META", "META")
add("Research apple.", "AAPL")
add("tell me about msft", "MSFT")
add("what about INTC", "INTC")
add("jpm please", "JPM")
add("show me DIS", "DIS")
add("amd stock", "AMD")
add("bac info", "BAC")


# ═══════════════════════════════════════════════════════════════════
# CATEGORY F-extra: Replacement samples for leaked ones  (~80 samples)
# F-补充类：替换泄漏样本的额外数据
# Purpose: Fill gaps left by leakage removal
# ═══════════════════════════════════════════════════════════════════

# ── Additional direct (replacing leaked direct samples) ──
add("What do you know about Apple stock?", "AAPL")
add("Show me Apple's latest numbers.", "AAPL")
add("I need a brief on Apple Inc.", "AAPL")
add("Microsoft stock quick look.", "MSFT")
add("Brief me on Microsoft's performance.", "MSFT")
add("How is Amazon's stock price trending?", "AMZN")
add("Amazon quarterly update please.", "AMZN")
add("Show me Amazon stock data.", "AMZN")
add("Summarize Tesla for me.", "TSLA")
add("I need Tesla's stock status.", "TSLA")
add("What's the deal with Netflix stock?", "NFLX")
add("Show me Netflix performance data.", "NFLX")
add("Brief me on JP Morgan stock.", "JPM")
add("How are JPM shares doing?", "JPM")
add("Show me Chase stock data.", "JPM")
add("JPMorgan Chase quarterly update.", "JPM")
add("How has Bank of America stock moved?", "BAC")
add("Show me BofA performance.", "BAC")
add("AMD stock status please.", "AMD")
add("How is Advanced Micro Devices performing?", "AMD")

# ── Additional Google/Alphabet class (replacing leaked class samples) ──
add("Show me Alphabet class A data.", "GOOGL")
add("Alphabet GOOGL performance report.", "GOOGL")
add("How have Google's A shares done?", "GOOGL")
add("I need class A Google stock info.", "GOOGL")
add("Update me on GOOGL.", "GOOGL")
add("Alphabet class C stock update.", "GOOG")
add("Show me GOOG performance.", "GOOG")
add("How has Google class C trended?", "GOOG")
add("I need the class C Alphabet data.", "GOOG")
add("GOOG quick look please.", "GOOG")
add("How is the Alphabet voting equity performing?", "GOOGL")

# ── Additional Berkshire (replacing leaked Berkshire samples) ──
add("Show me Berkshire Hathaway data.", "BRK-B")
add("Berkshire stock update please.", "BRK-B")
add("I need Berkshire Hathaway performance info.", "BRK-B")
add("How is Warren Buffett's holding company?", "BRK-B")
add("Update me on Berkshire Hathaway.", "BRK-B")
add("Show me BRK-B performance data.", "BRK-B")
add("How is the class B Berkshire doing?", "BRK-B")
add("What's happening with BRK/B stock?", "BRK-B")
add("BRK.B update please.", "BRK-B")
add("Show me BRK.A stock data.", "BRK-A")
add("How is BRK/A stock performing?", "BRK-A")
add("Update me on Berkshire class A.", "BRK-A")

# ── Additional META/Facebook (replacing leaked alias samples) ──
add("Show me Facebook stock data.", "META")
add("How has Facebook been doing lately?", "META")
add("Facebook shares update please.", "META")
add("Brief me on the Facebook stock.", "META")
add("How has the old Facebook stock moved?", "META")

# ── Additional colloquial (replacing leaked colloquial samples) ──
add("How is the company Bezos started?", "AMZN")
add("Show me Jeff Bezos' company stock.", "AMZN")
add("Research the high-end Berkshire shares.", "BRK-A")
add("How has the pricey Berkshire class done?", "BRK-A")

# ── Additional indirect / industry to improve coverage ──
add("How is the company that makes MacBooks?", "AAPL")
add("Research the company powering Siri.", "AAPL")
add("Analyze the tech firm behind Windows 11.", "MSFT")
add("How is the firm behind LinkedIn?", "MSFT")
add("Research the cloud giant behind S3 storage.", "AMZN")
add("Analyze the company behind Ring doorbells.", "AMZN")
add("How is the company behind Reels and Threads?", "META")
add("Research the VR headset company formerly called Facebook.", "META")
add("How is the company building the Cybertruck?", "TSLA")
add("Research the company behind Powerwall batteries.", "TSLA")
add("Analyze the firm powering AI data centers.", "NVDA")
add("How is the company behind the RTX graphics cards?", "NVDA")
add("Research the company behind Squid Game.", "NFLX")
add("Analyze the original streaming subscription service.", "NFLX")
add("Research the firm behind Disney World and Disneyland.", "DIS")
add("How is the company that makes the Marvel movies?", "DIS")
add("Analyze the foundry behind the M-series Apple chips.", "TSM")
add("Research the company behind the Tmall marketplace.", "BABA")
add("Analyze the stock behind the SHEIN competitor Temu.", "PDD")
add("Research the firm behind Foundry analytics software.", "PLTR")

# ═══════════════════════════════════════════════════════════════════
# CATEGORY G: Output Discipline / Edge Cases  (~25 samples)
# G 类：输出规范性 / 边界情况
# Purpose: Reinforce single-ticker-only output pattern
# ═══════════════════════════════════════════════════════════════════

# These are phrased in ways that might tempt the model to explain or refuse
add("What's the ticker for Apple?", "AAPL")
add("Which ticker is Google class C?", "GOOG")
add("What symbol should I use for Berkshire class B?", "BRK-B")
add("What is Facebook's current ticker?", "META")
add("Tell me the stock symbol for Tesla.", "TSLA")
add("What's the correct ticker for Netflix?", "NFLX")
add("Give me the symbol for Nvidia.", "NVDA")
add("Which ticker represents JPMorgan?", "JPM")
add("What is BofA's ticker symbol?", "BAC")
add("Which symbol do I use for Disney?", "DIS")
add("What ticker maps to Alibaba?", "BABA")
add("Give me the trading symbol for TSMC.", "TSM")
add("What's the ticker for AMD?", "AMD")
add("Which symbol is Intel traded under?", "INTC")
add("What's the stock symbol for Palantir?", "PLTR")
add("What is PDD Holdings' ticker?", "PDD")
add("What ticker should I use for Alphabet class A?", "GOOGL")
add("Which symbol represents Berkshire Hathaway class A?", "BRK-A")
add("Give me the ticker for Alphabet's non-voting stock.", "GOOG")
add("What's the current trading symbol for Facebook?", "META")
add("Which ticker is the voting Google class?", "GOOGL")
add("What symbol is Microsoft?", "MSFT")
add("What's the ticker for Amazon?", "AMZN")


# ═══════════════════════════════════════════════════════════════════
# DEV HARD SET (40 samples – same distribution as hard_eval, different sentences)
# 开发困难集（40 条 – 与 hard_eval 同分布但不同句子）
# ═══════════════════════════════════════════════════════════════════

dev_hard = [
    # ── CEO / founder references ──
    {"instruction": INSTRUCTION, "input": "How is the firm Tim Cook leads?", "output": "AAPL"},
    {"instruction": INSTRUCTION, "input": "Research Andy Jassy's tech empire.", "output": "AMZN"},
    {"instruction": INSTRUCTION, "input": "Analyze the company Mark Zuckerberg built.", "output": "META"},
    {"instruction": INSTRUCTION, "input": "How is Elon Musk's automotive venture?", "output": "TSLA"},
    {"instruction": INSTRUCTION, "input": "Research the firm Jensen Huang founded.", "output": "NVDA"},
    {"instruction": INSTRUCTION, "input": "Give me data on Nadella's software empire.", "output": "MSFT"},
    {"instruction": INSTRUCTION, "input": "How is the company Lisa Su turned around?", "output": "AMD"},
    {"instruction": INSTRUCTION, "input": "Research the conglomerate Buffett runs from Omaha.", "output": "BRK-B"},

    # ── Indirect class references (Google) ──
    {"instruction": INSTRUCTION, "input": "Research Google shares that don't carry voting power.", "output": "GOOG"},
    {"instruction": INSTRUCTION, "input": "How is the Alphabet class with shareholder votes?", "output": "GOOGL"},
    {"instruction": INSTRUCTION, "input": "Give me data on Google's governance-free share class.", "output": "GOOG"},
    {"instruction": INSTRUCTION, "input": "Analyze Alphabet's voting equity.", "output": "GOOGL"},
    {"instruction": INSTRUCTION, "input": "Research the Google equity with no board influence.", "output": "GOOG"},
    {"instruction": INSTRUCTION, "input": "How are the Alphabet shares with full voting participation?", "output": "GOOGL"},

    # ── Indirect class references (Berkshire) ──
    {"instruction": INSTRUCTION, "input": "Research Berkshire's broadly accessible shares.", "output": "BRK-B"},
    {"instruction": INSTRUCTION, "input": "How is the half-million-dollar Berkshire stock?", "output": "BRK-A"},
    {"instruction": INSTRUCTION, "input": "Give me data on the Berkshire class everyday investors buy.", "output": "BRK-B"},
    {"instruction": INSTRUCTION, "input": "Analyze Berkshire Hathaway's original undivided shares.", "output": "BRK-A"},
    {"instruction": INSTRUCTION, "input": "Research the Berkshire stock priced for retail buyers.", "output": "BRK-B"},
    {"instruction": INSTRUCTION, "input": "How is Berkshire's institutional-priced class?", "output": "BRK-A"},

    # ── Product / service references ──
    {"instruction": INSTRUCTION, "input": "Research the company that makes the iPad.", "output": "AAPL"},
    {"instruction": INSTRUCTION, "input": "How is the company behind Teams and Azure?", "output": "MSFT"},
    {"instruction": INSTRUCTION, "input": "Give me data on the owner of YouTube.", "output": "GOOGL"},
    {"instruction": INSTRUCTION, "input": "Analyze the company behind Prime Video.", "output": "AMZN"},
    {"instruction": INSTRUCTION, "input": "Research the company that owns ESPN and Pixar.", "output": "DIS"},
    {"instruction": INSTRUCTION, "input": "How is the firm behind the H100 chip?", "output": "NVDA"},
    {"instruction": INSTRUCTION, "input": "Give me info on the company behind the Model S.", "output": "TSLA"},
    {"instruction": INSTRUCTION, "input": "Analyze the creator of the Kindle.", "output": "AMZN"},
    {"instruction": INSTRUCTION, "input": "Research the company that runs chip foundry services for Apple.", "output": "TSM"},

    # ── Colloquial / nicknames ──
    {"instruction": INSTRUCTION, "input": "How is the entertainment empire behind Mickey Mouse?", "output": "DIS"},
    {"instruction": INSTRUCTION, "input": "Research the Omaha oracle's investment vehicle.", "output": "BRK-B"},
    {"instruction": INSTRUCTION, "input": "Give me data on the everything store stock.", "output": "AMZN"},
    {"instruction": INSTRUCTION, "input": "Analyze the electric car disruptor.", "output": "TSLA"},

    # ── Old name / rebrand ──
    {"instruction": INSTRUCTION, "input": "How is the platform that used to be called Facebook?", "output": "META"},
    {"instruction": INSTRUCTION, "input": "Research the social media company that rebranded to Meta.", "output": "META"},

    # ── Industry references ──
    {"instruction": INSTRUCTION, "input": "Give me data on the number one AI chip supplier.", "output": "NVDA"},
    {"instruction": INSTRUCTION, "input": "How is the dominant search engine company?", "output": "GOOGL"},
    {"instruction": INSTRUCTION, "input": "Research the largest contract chipmaker in the world.", "output": "TSM"},
    {"instruction": INSTRUCTION, "input": "Analyze the biggest e-commerce company from China.", "output": "BABA"},
    {"instruction": INSTRUCTION, "input": "How is the company behind the Temu shopping platform?", "output": "PDD"},
]


# ═══════════════════════════════════════════════════════════════════
# BUILD, VALIDATE, AND WRITE
# 构建、验证、写入
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    data_dir = pathlib.Path("data")

    print(f"Total v2 samples: {len(samples)}")

    # ── Dedup by input text ────────────────────────────
    seen = set()
    deduped = []
    dupes = 0
    for s in samples:
        key = s["input"].strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(s)
        else:
            dupes += 1
    samples = deduped
    print(f"After dedup: {len(samples)} (-{dupes} duplicates)")

    # ── Leakage check against frozen eval sets ─────────
    frozen_files = [
        "data/test.jsonl",
        "data/ambiguous_eval.jsonl",
        "data/hard_eval.jsonl",
    ]
    frozen_inputs = set()
    for fpath in frozen_files:
        try:
            for item in load_jsonl(fpath):
                frozen_inputs.add(item["input"].strip().lower())
        except FileNotFoundError:
            print(f"  Warning: {fpath} not found, skipping leakage check")

    leaked = [s for s in samples if s["input"].strip().lower() in frozen_inputs]
    if leaked:
        print(f"\n⚠  LEAKAGE DETECTED: {len(leaked)} training samples match frozen eval inputs!")
        for s in leaked:
            print(f"    LEAKED: \"{s['input']}\" → {s['output']}")
        # Remove leaked samples
        samples = [s for s in samples if s["input"].strip().lower() not in frozen_inputs]
        print(f"  After removing leaks: {len(samples)} samples")
    else:
        print("  ✓ No leakage against frozen eval sets")

    # Also check dev_hard for leakage
    dev_leaked = [s for s in dev_hard if s["input"].strip().lower() in frozen_inputs]
    if dev_leaked:
        print(f"\n⚠  DEV_HARD LEAKAGE: {len(dev_leaked)} dev_hard samples match frozen eval!")
        for s in dev_leaked:
            print(f"    LEAKED: \"{s['input']}\" → {s['output']}")
    else:
        print("  ✓ No dev_hard leakage against frozen eval sets")

    # ── Shuffle and split ──────────────────────────────
    random.shuffle(samples)
    n = len(samples)
    val_size = max(60, int(n * 0.15))  # ~15% for val, at least 60
    train = samples[:-val_size]
    val = samples[-val_size:]

    print(f"\nTrain v2: {len(train)} samples")
    print(f"Val v2:   {len(val)} samples")
    print(f"Dev hard: {len(dev_hard)} samples")

    # ── Write files ────────────────────────────────────
    write_jsonl(data_dir / "train_v2.jsonl", train)
    write_jsonl(data_dir / "val_v2.jsonl", val)
    write_jsonl(data_dir / "dev_hard.jsonl", dev_hard)

    # ── Print distributions ────────────────────────────
    print("\n── Train v2 label distribution ──")
    dist = Counter(s["output"] for s in train)
    for sym, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {sym}: {cnt}")

    print(f"\n── Val v2 label distribution ──")
    dist_v = Counter(s["output"] for s in val)
    for sym, cnt in sorted(dist_v.items(), key=lambda x: -x[1]):
        print(f"  {sym}: {cnt}")

    print(f"\n── Dev hard label distribution ──")
    dist_d = Counter(s["output"] for s in dev_hard)
    for sym, cnt in sorted(dist_d.items(), key=lambda x: -x[1]):
        print(f"  {sym}: {cnt}")

    print(f"\nDone. Files written to {data_dir}/")

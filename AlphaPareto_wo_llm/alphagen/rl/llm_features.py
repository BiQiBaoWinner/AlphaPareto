import os
from typing import List, Optional

import requests

_DEF_BASE = "http://localhost:11434/api/embeddings"  # Default Ollama Embeddings endpoint
_DEF_MODEL = "qwen-embed:4b"


def get_llm_embedding(
    pool_json: str,
    *,
    endpoint: Optional[str] = None,
    model: Optional[str] = None,
    expected_dim: Optional[int] = None,
    timeout: int = 6000,
    backend: Optional[str] = None,
) -> List[float]:
    """Call the local embedding HTTP API and return the raw embedding vector."""
    resolved_endpoint = endpoint or os.environ.get("OLLAMA_EMB_ENDPOINT", _DEF_BASE)
    resolved_model = model or os.environ.get("OLLAMA_EMB_MODEL", _DEF_MODEL)
    
    instruction = (
        "【Role Definition】\n"
        "You are the Chief Investment Officer of a quantitative hedge fund managing over $5 billion in assets, with 20+ years of factor research experience. "
        "You excel at decoding market microstructure signals from mathematical expressions, precisely identifying the economic logic behind factors "
        "(e.g., liquidity premiums, behavioral bias arbitrage, volatility surface arbitrage), and evaluating their risk-return contributions in portfolios.\n\n"
        
        "【Core Task】\n"
        "Deeply internalize the mathematical structure and portfolio characteristics of the following factor pool. Your analysis must strictly derive from the JSON data—"
        "no external assumptions permitted. Focus on:\n"
        "1) **Factor Style Deconstruction**: Map each expression to professional factor taxonomies (e.g., Barra style factors/alternative data factors/tail risk factors)\n"
        "2) **Information Topology Analysis**: Detect data source blind spots (e.g., over-reliance on raw prices? missing volume-price interactions? ignoring volatility surfaces?)\n"
        "3) **Risk Heatmap**: Identify vulnerabilities exposed by weight configuration (e.g., negative weight concentration, extreme value sensitivity, expression homogenization risk)\n"
        "4) **Evolution Path Design**: Conceive next-generation factors that penetrate current information blind spots (specify mathematical operations and data types)\n\n"
        
        "【Professional Analysis Framework】\n"
        "Apply the following quantitative expertise for deep decoding:\n"
        "- **Price Transformation Layer**: Identify market state assumptions implied by operations like Log/EMA/Greater/Less (e.g., Log($price) assumes log-normal distribution)\n"
        "- **Time Scale Layer**: Parse window parameters (40d/20d/10d) to determine market cycle sensitivity (short-term sentiment vs. long-term trends)\n"
        "- **Nonlinearity Layer**: Detect strategy logic formed by combinations of Add/Sub/Mul/Div with conditional functions (Greater/Less) (e.g., volatility breakout strategies)\n"
        "- **Weight Diagnostics**: Calculate weight entropy to assess diversification; flag factors with |weight|>0.2 as risk concentration points\n"
        "- **Data Source Audit**: Count frequencies of $high/$low/$close/$open/$vwap/$volume appearances to evaluate price dimension coverage completeness\n\n"
        
        "【Critical Example Anchoring】\n"
        "Below is the typical structure of the factor pool you will analyze (note expression complexity and weight distribution characteristics):\n"
        "exprs = [\n"
        "  'Sub(Mul(EMA(Log(Sum(Add(0.5,Div($high,0.01)),40d)),40d),-10.0),-0.01)',  # Long-cycle log-price trend factor\n"
        "  'Add(Div(Less(-30.0,$close),Add(30.0,Greater($low,-30.0))),Greater(Log($open),10.0))',  # Extreme price conditional factor\n"
        "  'Sub(Less(Greater(Add(5.0,Add($close,-30.0)),-30.0),-10.0),-1.0)',  # Price range oscillation factor\n"
        "  'Div(2.0,$vwap)',  # VWAP reversal factor\n"
        "  'Sub(Div(5.0,$high),5.0)',  # High-price suppression factor\n"
        "  'Add(Div(-10.0,$low),-5.0)',  # Low-price enhancement factor\n"
        "  'Add(Sub(Abs(Log(Sub(Div(Greater(-5.0,Log($close)),0.01),10.0))),-0.5),-30.0)',  # Volatility nonlinearity factor\n"
        "  'Greater(Div(EMA(Div(Greater(Log(Div(0.01,$high)),-30.0),-10.0),20d),$low),-10.0)',  # Multi-scale volatility factor\n"
        "  'Log(Sub(Greater(-30.0,$vwap),Div(Div($volume,$open),-30.0)))',  # Volume-price divergence factor\n"
        "  'Ref(Log(Greater(Mul(-5.0,Mean(Mul($open,-0.5),10d)),Less(-30.0,$open))),20d)'  # Lagged open-price sentiment factor\n"
        "],\n"
        "weights = [\n"
        "  -0.0732,  # Mild negative exposure\n"
        "  -0.2274,  # Significant negative exposure (risk concentration point)\n"
        "  -0.0497,  \n"
        "   0.0456,  \n"
        "   0.2256,  # Core positive exposure\n"
        "  -0.0793,  \n"
        "  -0.1027,  \n"
        "  -0.4176,  # Severe risk concentration (41.76% of total weight magnitude)\n"
        "  -0.1442,  \n"
        "   0.0939   \n"
        "],\n"
        ">>> This example reveals critical risk: Factor #8 holds 41.76% absolute weight concentration with triple-nested nonlinear operations, highly vulnerable during market regime shifts\n\n"
        
        "【Final Output Directive】\n"
        "Now, channel your professional cognition entirely onto the real-world factor pool described in the JSON below.\n"
        "**OMIT all intermediate analysis steps, background context, and pleasantries.**\n"
        "Directly based on your analysis of current portfolio blind spots and risks, output **3** construction style recommendations for the \"Next Factor,\" ranked by priority (High to Low).\n\n"
        "Output format must strictly adhere to the following (Style Name MUST be selected from the provided list):\n"
        "Allowed Style List: [Trend Momentum], [Mean Reversion], [Volatility Structure], [Price-Volume Interaction], [Liquidity Premium], [Market Sentiment], [Tail Risk], [Correlation Structure], [Higher-Moment Features], [Term Structure], [Asymmetry]\n\n"
        "1. **[Selected Style Name]**: <Specific mathematical construction proposal, including recommended operators and data sources> —— <Expected contribution to filling current blind spots or reducing correlation>\n"
        "2. **[Selected Style Name]**: ...\n"
        "3. **[Selected Style Name]**: ...\n\n"
        
        "【Cognitive Activation】\n"
        "Begin decoding and output recommendations directly:"
    )

    composed_prompt = f"{instruction}\n\nJSON:\n{pool_json}"

    payload = {"model": resolved_model, "prompt": composed_prompt}
    headers = {"Content-Type": "application/json"}

    resp = requests.post(resolved_endpoint, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()

    data = resp.json()

    # {"embedding": [...], "num_tokens": ...}
    embedding = data.get("embedding")
    if embedding is None:
        raise RuntimeError(f"Embedding response invalid: {data}")

    vec = [float(v) for v in embedding]

    if expected_dim is not None and len(vec) != expected_dim:
        raise ValueError(f"Embedding dim {len(vec)} != expected {expected_dim}")

    return vec
import streamlit as st


def apply_global_style():
    st.markdown(
        """
        <style>
        /* ===== GOOGLE FONTS ===== */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: #0B0F19;
            color: #E2E8F0;
        }

        /* ===== HERO HEADER ===== */
        .hero-banner {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 36px 32px;
            margin-bottom: 28px;
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(12px);
        }

        .hero-title {
            font-size: 2.3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 50%, #38BDF8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            letter-spacing: -0.02em;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            color: #94A3B8;
            max-width: 800px;
            line-height: 1.6;
            margin-bottom: 18px;
        }

        .badge-container {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.03em;
        }

        .badge-blue {
            background: rgba(56, 189, 248, 0.12);
            color: #38BDF8;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }

        .badge-purple {
            background: rgba(168, 85, 247, 0.12);
            color: #C084FC;
            border: 1px solid rgba(168, 85, 247, 0.3);
        }

        .badge-emerald {
            background: rgba(52, 211, 153, 0.12);
            color: #34D399;
            border: 1px solid rgba(52, 211, 153, 0.3);
        }

        /* ===== CARDS & CONTAINERS ===== */
        .glass-card {
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(8px);
            transition: border-color 0.2s ease;
        }

        .glass-card:hover {
            border-color: rgba(56, 189, 248, 0.3);
        }

        .metric-badge {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
        }

        .metric-val {
            font-size: 1.8rem;
            font-weight: 700;
            color: #F8FAFC;
        }

        .metric-label {
            font-size: 0.8rem;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 4px;
        }

        /* ===== TABS STYLING ===== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: rgba(15, 23, 42, 0.6);
            padding: 8px;
            border-radius: 14px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 8px 18px;
            color: #94A3B8;
            font-weight: 600;
            font-size: 0.92rem;
            transition: all 0.2s ease;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%) !important;
            color: #38BDF8 !important;
            border: 1px solid rgba(56, 189, 248, 0.4) !important;
        }

        /* ===== BUTTONS ===== */
        .stButton>button {
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.2s ease;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .stButton>button[kind="primary"] {
            background: linear-gradient(135deg, #0284C7 0%, #4F46E5 100%);
            border: none;
            box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4);
        }

        .stButton>button[kind="primary"]:hover {
            box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6);
            transform: translateY(-1px);
        }

        /* ===== CODE BLOCKS ===== */
        code, pre {
            font-family: 'JetBrains Mono', monospace !important;
            border-radius: 8px !important;
        }

        /* ===== SECTION SEPARATORS ===== */
        .section-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-top: 10px;
            margin-bottom: 18px;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 700;
            color: #F1F5F9;
            margin: 0;
        }

        .section-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 34px;
            height: 34px;
            background: rgba(56, 189, 248, 0.15);
            border-radius: 8px;
            color: #38BDF8;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

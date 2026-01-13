"""
Constants used for manufacturer and SDS search tools.
"""

# Domains to penalize during manufacturer search, as they are typically not
# official manufacturer sites (e.g., social media, news, directories).
BAD_DOMAINS = {
    "wikipedia.org",
    "linkedin.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "bloomberg.com",
    "yahoo.com",
    "crunchbase.com",
    "zoominfo.com",
}

# Common corporate suffixes to remove from names for better search tokenization.
CORP_STOPWORDS = {
    "inc",
    "incorporated",
    "llc",
    "ltd",
    "limited",
    "co",
    "company",
    "corp",
    "corporation",
    "gmbh",
    "ag",
    "plc",
    "srl",
    "sa",
    "bv",
    "nv",
    "kg",
    "oy",
    "ab",
    "sas",
    "pty",
    "pte",
    "sro",
    "kk",
}

# A tuple of top-level domains specific to Australia, used to prioritize
# AU-based company websites.
AU_TLDS = (
    ".com.au",
    ".net.au",
    ".org.au",
    ".edu.au",
    ".gov.au",
    ".asn.au",
    ".id.au",
    ".au",
)

# A list of search query templates to find a manufacturer's website.
# The `{name}` placeholder will be filled with the manufacturer's name.
INTENT_QUERIES = [
    '"{name}" "contact us" Australia',
    '"{name}" contact Australia',
    '"{name}" "about us" Australia',
    '"{name}" website Australia',
    '"{name}" ".com.au"',
]

# Keywords used to identify Safety Data Sheet (SDS) documents in search results.
SDS_KEYWORDS = ("sds", "safety data sheet", "safety datasheet")

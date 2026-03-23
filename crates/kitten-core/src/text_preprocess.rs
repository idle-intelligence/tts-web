/// Text preprocessor for KittenTTS.
///
/// Converts currencies, units, scale suffixes, fractions, scientific notation,
/// IP addresses, ranges, and other patterns into spoken-word form before
/// passing text to espeak-ng.
///
/// espeak-ng already handles: plain integers, ordinals (1st/2nd), times (3:45).
/// This module handles everything else.

// ---------------------------------------------------------------------------
// Number-to-words
// ---------------------------------------------------------------------------

fn ones(n: u64) -> &'static str {
    match n {
        0 => "zero", 1 => "one", 2 => "two", 3 => "three", 4 => "four",
        5 => "five", 6 => "six", 7 => "seven", 8 => "eight", 9 => "nine",
        10 => "ten", 11 => "eleven", 12 => "twelve", 13 => "thirteen",
        14 => "fourteen", 15 => "fifteen", 16 => "sixteen", 17 => "seventeen",
        18 => "eighteen", 19 => "nineteen",
        _ => "",
    }
}

fn tens_word(n: u64) -> &'static str {
    match n {
        2 => "twenty", 3 => "thirty", 4 => "forty", 5 => "fifty",
        6 => "sixty", 7 => "seventy", 8 => "eighty", 9 => "ninety",
        _ => "",
    }
}

/// Convert a non-negative integer up to 999_999_999_999 into words.
pub fn number_to_words(n: i64) -> String {
    if n < 0 {
        return format!("negative {}", number_to_words(-n));
    }
    let n = n as u64;
    if n == 0 {
        return "zero".to_string();
    }
    number_to_words_u64(n)
}

fn number_to_words_u64(n: u64) -> String {
    if n == 0 {
        return "zero".to_string();
    }
    if n < 20 {
        return ones(n).to_string();
    }
    if n < 100 {
        let t = tens_word(n / 10);
        let r = n % 10;
        if r == 0 { t.to_string() } else { format!("{} {}", t, ones(r)) }
    } else if n < 1_000 {
        let h = n / 100;
        let r = n % 100;
        let s = format!("{} hundred", ones(h));
        if r == 0 { s } else { format!("{} {}", s, number_to_words_u64(r)) }
    } else if n < 1_000_000 {
        let hi = n / 1_000;
        let lo = n % 1_000;
        let s = format!("{} thousand", number_to_words_u64(hi));
        if lo == 0 { s } else { format!("{} {}", s, number_to_words_u64(lo)) }
    } else if n < 1_000_000_000 {
        let hi = n / 1_000_000;
        let lo = n % 1_000_000;
        let s = format!("{} million", number_to_words_u64(hi));
        if lo == 0 { s } else { format!("{} {}", s, number_to_words_u64(lo)) }
    } else if n < 1_000_000_000_000 {
        let hi = n / 1_000_000_000;
        let lo = n % 1_000_000_000;
        let s = format!("{} billion", number_to_words_u64(hi));
        if lo == 0 { s } else { format!("{} {}", s, number_to_words_u64(lo)) }
    } else {
        let hi = n / 1_000_000_000_000;
        let lo = n % 1_000_000_000_000;
        let s = format!("{} trillion", number_to_words_u64(hi));
        if lo == 0 { s } else { format!("{} {}", s, number_to_words_u64(lo)) }
    }
}

/// Parse an unsigned integer from s[start..], returning (value, end_index).
fn parse_uint(s: &[u8], start: usize) -> Option<(u64, usize)> {
    let mut i = start;
    while i < s.len() && s[i].is_ascii_digit() {
        i += 1;
    }
    if i == start { return None; }
    let num_str = std::str::from_utf8(&s[start..i]).ok()?;
    let v: u64 = num_str.parse().ok()?;
    Some((v, i))
}

/// Convert decimal digits after decimal point to spoken form, digit by digit.
/// "5" → "five", "50" → "five zero", "05" → "zero five"
fn decimal_digits_to_words(digits: &str) -> String {
    digits.chars()
        .map(|c| ones(c as u64 - '0' as u64))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Expand a float string like "50.99" into words.
/// Integer part via number_to_words, decimal part digit by digit.
fn float_to_words(int_part: u64, frac_str: &str) -> String {
    let int_words = number_to_words_u64(int_part);
    if frac_str.is_empty() {
        int_words
    } else {
        format!("{} point {}", int_words, decimal_digits_to_words(frac_str))
    }
}

// ---------------------------------------------------------------------------
// Token classification helpers
// ---------------------------------------------------------------------------

fn is_digit(c: u8) -> bool { c.is_ascii_digit() }

/// Try to parse a leading number (int or float) from a byte slice.
/// Returns (integer_part, optional_fraction_str, end_index).
fn parse_number_prefix(s: &[u8], start: usize) -> Option<(u64, String, usize)> {
    let (int_val, mut i) = parse_uint(s, start)?;
    let mut frac = String::new();
    // Check for decimal point followed by digits
    if i < s.len() && s[i] == b'.' {
        let j = i + 1;
        if j < s.len() && is_digit(s[j]) {
            let (_, k) = parse_uint(s, j)?;
            frac = std::str::from_utf8(&s[j..k]).unwrap_or("").to_string();
            i = k;
        }
    }
    Some((int_val, frac, i))
}

// ---------------------------------------------------------------------------
// Currency
// ---------------------------------------------------------------------------

struct Currency {
    unit: &'static str,
    cents: &'static str,
    invariant: bool, // true if unit doesn't pluralize (e.g. yen)
}

fn currency_for(c: char) -> Option<Currency> {
    match c {
        '$' => Some(Currency { unit: "dollar",  cents: "cent",  invariant: false }),
        '€' => Some(Currency { unit: "euro",    cents: "cent",  invariant: false }),
        '£' => Some(Currency { unit: "pound",   cents: "penny", invariant: false }),
        '¥' => Some(Currency { unit: "yen",     cents: "",      invariant: true  }),
        _ => None,
    }
}

fn pluralize(word: &str, n: u64) -> String {
    if n == 1 { word.to_string() } else { format!("{}s", word) }
}

/// Try to expand a currency token starting at `pos`.
/// Returns (expanded_string, bytes_consumed) or None.
fn try_expand_currency(s: &[u8], pos: usize) -> Option<(String, usize)> {
    if pos >= s.len() { return None; }
    // Decode the full Unicode char (currency symbols can be multi-byte UTF-8)
    let tail_str = std::str::from_utf8(&s[pos..]).ok()?;
    let c = tail_str.chars().next()?;
    let cur = currency_for(c)?;
    let prefix_len = c.len_utf8();
    let (int_val, frac, end) = parse_number_prefix(s, pos + prefix_len)?;
    if end == pos + prefix_len { return None; } // no digits

    // Check for scale suffix immediately after the number (e.g. $2.5B → "two point five billion dollars")
    if end < s.len() {
        let scale = match s[end] {
            b'T' | b't' if end + 1 >= s.len() || !s[end+1].is_ascii_alphanumeric() => Some(("trillion", end + 1)),
            b'B' | b'b' if end + 1 >= s.len() || !s[end+1].is_ascii_alphanumeric() => Some(("billion", end + 1)),
            b'M'        if end + 1 >= s.len() || !s[end+1].is_ascii_alphanumeric() => Some(("million", end + 1)),
            b'K' | b'k' if end + 1 >= s.len() || !s[end+1].is_ascii_alphanumeric() => Some(("thousand", end + 1)),
            _ => None,
        };
        if let Some((scale_word, end2)) = scale {
            let num_words = float_to_words(int_val, &frac);
            let unit_str = pluralize(cur.unit, 0); // plural for scale amounts
            return Some((format!("{} {} {}", num_words, scale_word, unit_str), end2));
        }
    }

    let unit_str = if cur.invariant { cur.unit.to_string() } else { pluralize(cur.unit, int_val) };
    let mut result = format!("{} {}", number_to_words_u64(int_val), unit_str);

    // Cents from fraction (only if frac has exactly 2 digits)
    if !frac.is_empty() && !cur.cents.is_empty() {
        let cent_val: u64 = if frac.len() == 1 {
            frac.parse::<u64>().unwrap_or(0) * 10
        } else if frac.len() == 2 {
            frac.parse::<u64>().unwrap_or(0)
        } else {
            // ignore sub-cent fractions, just say "point X"
            return Some((
                format!("{} point {} {}", number_to_words_u64(int_val), decimal_digits_to_words(&frac), pluralize(cur.unit, 0)),
                end,
            ));
        };
        if cent_val > 0 {
            result = format!("{} and {} {}", result, number_to_words_u64(cent_val), pluralize(cur.cents, cent_val));
        }
    } else if !frac.is_empty() && cur.cents.is_empty() {
        // Yen doesn't have cents, just say "point N"
        result = format!("{} point {}", result, decimal_digits_to_words(&frac));
    }

    Some((result, end))
}

// ---------------------------------------------------------------------------
// Units
// ---------------------------------------------------------------------------

/// Returns (unit_suffix_bytes, spoken_form) for known unit suffixes.
/// Longer suffixes first so greedy matching works.
fn unit_suffix(s: &[u8], pos: usize) -> Option<(usize, &'static str)> {
    let tail = &s[pos..];
    // Case-sensitive matching for units — common abbreviations
    let units: &[(&[u8], &str)] = &[
        // Distance
        (b"km",  "kilometers"),
        (b"cm",  "centimeters"),
        (b"mm",  "millimeters"),
        (b"mi",  "miles"),
        (b"ft",  "feet"),
        (b"in",  "inches"),
        (b"m",   "meters"),
        // Mass
        (b"kg",  "kilograms"),
        (b"mg",  "milligrams"),
        (b"lb",  "pounds"),
        (b"lbs", "pounds"),
        (b"oz",  "ounces"),
        (b"g",   "grams"),
        // Speed
        (b"mph", "miles per hour"),
        (b"kph", "kilometers per hour"),
        // Data
        (b"GB",  "gigabytes"),
        (b"MB",  "megabytes"),
        (b"KB",  "kilobytes"),
        (b"TB",  "terabytes"),
        (b"GHz", "gigahertz"),
        (b"MHz", "megahertz"),
        (b"kHz", "kilohertz"),
        (b"Hz",  "hertz"),
        // Power/Voltage
        (b"kW",  "kilowatts"),
        (b"MW",  "megawatts"),
        (b"W",   "watts"),
        (b"kV",  "kilovolts"),
        (b"mV",  "millivolts"),
        (b"V",   "volts"),
        // Time (espeak handles plain numbers but not with suffixes)
        (b"ms",  "milliseconds"),
        (b"ns",  "nanoseconds"),
        (b"us",  "microseconds"),
        (b"s",   "seconds"),
        // Temperature (degrees handled separately for °F/°C/°K)
        // Area/Volume
        (b"km2", "square kilometers"),
        (b"m2",  "square meters"),
        (b"L",   "liters"),
        (b"mL",  "milliliters"),
        (b"ml",  "milliliters"),
    ];
    for (suffix, spoken) in units {
        if tail.starts_with(suffix) {
            // Make sure the unit isn't followed by more alphanumeric chars (avoid false matches)
            let after = pos + suffix.len();
            if after >= s.len() || !s[after].is_ascii_alphanumeric() {
                return Some((suffix.len(), spoken));
            }
        }
    }
    None
}

/// Try to expand a temperature token like "72°F".
fn try_expand_temperature(s: &[u8], pos: usize) -> Option<(String, usize)> {
    let (int_val, frac, end) = parse_number_prefix(s, pos)?;
    if end >= s.len() || s[end] != b'\xc2' { return None; } // UTF-8 first byte of °
    // ° is U+00B0, encoded as 0xC2 0xB0
    if end + 1 >= s.len() || s[end + 1] != b'\xb0' { return None; }
    let after_deg = end + 2;
    let scale = if after_deg < s.len() {
        match s[after_deg] {
            b'F' => Some(("fahrenheit", after_deg + 1)),
            b'C' => Some(("celsius", after_deg + 1)),
            b'K' => Some(("kelvin", after_deg + 1)),
            _ => None,
        }
    } else {
        None
    };
    let num_words = float_to_words(int_val, &frac);
    let (scale_word, end2) = scale.unwrap_or(("degrees", after_deg));
    let text = if scale_word == "fahrenheit" || scale_word == "celsius" || scale_word == "kelvin" {
        format!("{} degrees {}", num_words, scale_word)
    } else {
        format!("{} degrees", num_words)
    };
    Some((text, end2))
}

// ---------------------------------------------------------------------------
// Scale suffixes: 7B, 2.5M, 1K, 3T
// ---------------------------------------------------------------------------

fn try_expand_scale(s: &[u8], pos: usize) -> Option<(String, usize)> {
    let (int_val, frac, end) = parse_number_prefix(s, pos)?;
    if end >= s.len() { return None; }
    let scale = match s[end] {
        b'T' | b't' if end + 1 >= s.len() || !s[end+1].is_ascii_alphanumeric() => "trillion",
        b'B' | b'b' if end + 1 >= s.len() || !s[end+1].is_ascii_alphanumeric() => "billion",
        b'M'        if end + 1 >= s.len() || !s[end+1].is_ascii_alphanumeric() => "million",
        b'K' | b'k' if end + 1 >= s.len() || !s[end+1].is_ascii_alphanumeric() => "thousand",
        _ => return None,
    };
    let num_words = float_to_words(int_val, &frac);
    Some((format!("{} {}", num_words, scale), end + 1))
}

// ---------------------------------------------------------------------------
// Percentage
// ---------------------------------------------------------------------------

fn try_expand_percent(s: &[u8], pos: usize) -> Option<(String, usize)> {
    let (int_val, frac, end) = parse_number_prefix(s, pos)?;
    if end >= s.len() || s[end] != b'%' { return None; }
    let num_words = float_to_words(int_val, &frac);
    Some((format!("{} percent", num_words), end + 1))
}

// ---------------------------------------------------------------------------
// Fraction: 1/2, 3/4, 2/3
// ---------------------------------------------------------------------------

fn fraction_word(num: u64, den: u64) -> Option<String> {
    let n_words = number_to_words_u64(num);
    let d_word = match den {
        2 => if num == 1 { "half" } else { "halves" },
        3 => if num == 1 { "third" } else { "thirds" },
        4 => if num == 1 { "quarter" } else { "quarters" },
        5 => if num == 1 { "fifth" } else { "fifths" },
        6 => if num == 1 { "sixth" } else { "sixths" },
        7 => if num == 1 { "seventh" } else { "sevenths" },
        8 => if num == 1 { "eighth" } else { "eighths" },
        9 => if num == 1 { "ninth" } else { "ninths" },
        10 => if num == 1 { "tenth" } else { "tenths" },
        _ => return None,
    };
    Some(format!("{} {}", n_words, d_word))
}

fn try_expand_fraction(s: &[u8], pos: usize) -> Option<(String, usize)> {
    let (num, end_num) = parse_uint(s, pos)?;
    if end_num >= s.len() || s[end_num] != b'/' { return None; }
    let (den, end_den) = parse_uint(s, end_num + 1)?;
    if den == 0 { return None; }
    // Make sure it's not followed by more digits (avoid matching e.g. version numbers)
    if end_den < s.len() && is_digit(s[end_den]) { return None; }
    // Use named fraction if available, otherwise "N over M"
    let result = fraction_word(num, den)
        .unwrap_or_else(|| format!("{} over {}", number_to_words_u64(num), number_to_words_u64(den)));
    Some((result, end_den))
}

// ---------------------------------------------------------------------------
// Scientific notation: 1e4, 1e-4, 2.5e10
// ---------------------------------------------------------------------------

fn try_expand_scientific(s: &[u8], pos: usize) -> Option<(String, usize)> {
    let (int_val, frac, end) = parse_number_prefix(s, pos)?;
    if end >= s.len() || (s[end] != b'e' && s[end] != b'E') { return None; }
    let mut i = end + 1;
    let negative = i < s.len() && s[i] == b'-';
    let positive = i < s.len() && s[i] == b'+';
    if negative || positive { i += 1; }
    let (exp_val, end_exp) = parse_uint(s, i)?;
    if end_exp == i { return None; } // no exponent digits

    let mantissa = float_to_words(int_val, &frac);
    let exp_words = number_to_words_u64(exp_val);
    let sign = if negative { "negative " } else { "" };
    Some((format!("{} times ten to the {}{}", mantissa, sign, exp_words), end_exp))
}

// ---------------------------------------------------------------------------
// IP address: 192.168.1.1
// ---------------------------------------------------------------------------

fn digits_to_spoken(s: &[u8], start: usize, end: usize) -> String {
    s[start..end].iter()
        .map(|&b| ones((b - b'0') as u64))
        .collect::<Vec<_>>()
        .join(" ")
}

fn try_expand_ip(s: &[u8], pos: usize) -> Option<(String, usize)> {
    // Must match: digits.digits.digits.digits (all segments 0-255)
    // Each octet spoken digit by digit: 192 → "one nine two"
    let mut parts: Vec<String> = Vec::with_capacity(4);
    let mut i = pos;
    for seg in 0..4 {
        let start = i;
        let (val, end) = parse_uint(s, i)?;
        if val > 255 { return None; }
        parts.push(digits_to_spoken(s, start, end));
        i = end;
        if seg < 3 {
            if i >= s.len() || s[i] != b'.' { return None; }
            i += 1;
        }
    }
    // Must not be followed by more digits or dots (would be a longer number)
    if i < s.len() && (is_digit(s[i]) || s[i] == b'.') { return None; }
    Some((parts.join(" dot "), i))
}

// ---------------------------------------------------------------------------
// Range: 10-20, 100-200
// ---------------------------------------------------------------------------

fn try_expand_range(s: &[u8], pos: usize) -> Option<(String, usize)> {
    let (a, end_a) = parse_uint(s, pos)?;
    if end_a >= s.len() || s[end_a] != b'-' { return None; }
    let (b, end_b) = parse_uint(s, end_a + 1)?;
    if end_b == end_a + 1 { return None; } // no second number
    // Not followed by more word chars (avoid matching e.g. "10-point")
    if end_b < s.len() && (s[end_b].is_ascii_alphabetic() || s[end_b] == b'-') { return None; }
    Some((format!("{} to {}", number_to_words_u64(a), number_to_words_u64(b)), end_b))
}

// ---------------------------------------------------------------------------
// Contractions
// ---------------------------------------------------------------------------

fn expand_contractions(s: &str) -> String {
    // Process word by word to avoid partial matches
    // We scan for apostrophes and expand known patterns
    let contractions: &[(&str, &str)] = &[
        ("won't", "will not"),
        ("can't", "cannot"),
        ("don't", "do not"),
        ("doesn't", "does not"),
        ("didn't", "did not"),
        ("isn't", "is not"),
        ("aren't", "are not"),
        ("wasn't", "was not"),
        ("weren't", "were not"),
        ("hasn't", "has not"),
        ("haven't", "have not"),
        ("hadn't", "had not"),
        ("wouldn't", "would not"),
        ("shouldn't", "should not"),
        ("couldn't", "could not"),
        ("mustn't", "must not"),
        ("mightn't", "might not"),
        ("needn't", "need not"),
        ("I'm", "I am"),
        ("I've", "I have"),
        ("I'll", "I will"),
        ("I'd", "I would"),
        ("you're", "you are"),
        ("you've", "you have"),
        ("you'll", "you will"),
        ("you'd", "you would"),
        ("he's", "he is"),
        ("he'll", "he will"),
        ("he'd", "he would"),
        ("she's", "she is"),
        ("she'll", "she will"),
        ("she'd", "she would"),
        ("it's", "it is"),
        ("it'll", "it will"),
        ("we're", "we are"),
        ("we've", "we have"),
        ("we'll", "we will"),
        ("we'd", "we would"),
        ("they're", "they are"),
        ("they've", "they have"),
        ("they'll", "they will"),
        ("they'd", "they would"),
        ("that's", "that is"),
        ("that'll", "that will"),
        ("there's", "there is"),
        ("there're", "there are"),
        ("what's", "what is"),
        ("what're", "what are"),
        ("what'll", "what will"),
        ("who's", "who is"),
        ("who'll", "who will"),
        ("let's", "let us"),
        ("'cause", "because"),
    ];

    let mut result = s.to_string();
    for (from, to) in contractions {
        // Case-insensitive replacement: try exact, then lower-cased token
        // Simple approach: replace all occurrences (word-boundary aware via surrounding chars)
        result = replace_word_ci(&result, from, to);
    }
    result
}

/// Replace `pattern` with `replacement` in `s`, case-insensitively for the pattern,
/// but only when surrounded by non-alphanumeric chars (word boundary).
fn replace_word_ci(s: &str, pattern: &str, replacement: &str) -> String {
    let lower_s = s.to_lowercase();
    let lower_p = pattern.to_lowercase();
    let mut out = String::with_capacity(s.len());
    let mut last = 0;
    let mut search_from = 0;
    while let Some(idx) = lower_s[search_from..].find(&lower_p) {
        let start = search_from + idx;
        let end = start + pattern.len();
        // Word boundary check
        let before_ok = start == 0 || !s.as_bytes()[start - 1].is_ascii_alphanumeric();
        let after_ok = end >= s.len() || !s.as_bytes()[end].is_ascii_alphanumeric();
        if before_ok && after_ok {
            out.push_str(&s[last..start]);
            out.push_str(replacement);
            last = end;
            search_from = end;
        } else {
            search_from = start + 1;
        }
    }
    out.push_str(&s[last..]);
    out
}

// ---------------------------------------------------------------------------
// URL removal
// ---------------------------------------------------------------------------

fn remove_urls(s: &str) -> String {
    // Split on whitespace, drop tokens that look like URLs
    s.split_whitespace()
        .filter(|tok| !tok.starts_with("http://") && !tok.starts_with("https://"))
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// Whitespace normalisation
// ---------------------------------------------------------------------------

fn normalize_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_space = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !in_space { out.push(' '); }
            in_space = true;
        } else {
            out.push(c);
            in_space = false;
        }
    }
    out.trim().to_string()
}

// ---------------------------------------------------------------------------
// Token scanner
// ---------------------------------------------------------------------------

/// Scan the byte slice and expand numeric patterns, returning the result as
/// a String.  Non-numeric characters are passed through unchanged.
fn expand_tokens(s: &str) -> String {
    let bytes = s.as_bytes();
    let len = bytes.len();
    let mut out = String::with_capacity(len * 2);
    let mut i = 0;

    while i < len {
        // Skip URLs (already removed, but just in case)
        // Try each pattern in priority order

        // Currency: starts with currency symbol (handle multi-byte UTF-8)
        if let Some(cur) = {
            // Decode the Unicode char at position i
            let sc = std::str::from_utf8(&bytes[i..]).ok().and_then(|tail| tail.chars().next());
            if let Some(sc) = sc {
                if currency_for(sc).is_some() {
                    try_expand_currency(bytes, i)
                } else { None }
            } else { None }
        } {
            out.push_str(&cur.0);
            i = cur.1;
            continue;
        }

        // Temperature: digits followed by °
        if is_digit(bytes[i]) {
            // Check for IP address first (most specific digit pattern)
            if let Some((expanded, end)) = try_expand_ip(bytes, i) {
                out.push_str(&expanded);
                i = end;
                continue;
            }

            // Scientific notation
            if let Some((expanded, end)) = try_expand_scientific(bytes, i) {
                out.push_str(&expanded);
                i = end;
                continue;
            }

            // Fraction
            if let Some((expanded, end)) = try_expand_fraction(bytes, i) {
                out.push_str(&expanded);
                i = end;
                continue;
            }

            // Range (digits-digits)
            if let Some((expanded, end)) = try_expand_range(bytes, i) {
                out.push_str(&expanded);
                i = end;
                continue;
            }

            // Temperature
            if let Some((expanded, end)) = try_expand_temperature(bytes, i) {
                out.push_str(&expanded);
                i = end;
                continue;
            }

            // Percentage
            if let Some((expanded, end)) = try_expand_percent(bytes, i) {
                out.push_str(&expanded);
                i = end;
                continue;
            }

            // Scale suffix (7B, 2.5M, 1K, 3T) — only match if next char is alphanumeric scale letter
            // We try this before plain units to catch e.g. "7B" vs "7m" (meters)
            if let Some((expanded, end)) = try_expand_scale(bytes, i) {
                out.push_str(&expanded);
                i = end;
                continue;
            }

            // Units (e.g. 100km)
            if let Some((int_val, frac, end)) = parse_number_prefix(bytes, i) {
                if let Some((unit_len, unit_word)) = unit_suffix(bytes, end) {
                    let num_words = float_to_words(int_val, &frac);
                    out.push_str(&format!("{} {}", num_words, unit_word));
                    i = end + unit_len;
                    continue;
                }
            }

            // Plain number — leave for espeak (it handles integers fine)
            // But we still need to consume the digits and any decimal
            // to avoid re-triggering patterns inside a number.
            // Actually, just pass through — espeak handles it.
        }

        // Pass current byte through
        // For multi-byte UTF-8, copy the whole code point
        let c = s[i..].chars().next().unwrap_or('\0');
        out.push(c);
        i += c.len_utf8();
    }

    out
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Preprocess text for KittenTTS.
///
/// Applies (in order):
/// 1. URL removal
/// 2. Contraction expansion
/// 3. Token-level numeric pattern expansion (currencies, units, scale, %, fractions, sci, IP, ranges)
/// 4. Lowercase
/// 5. Whitespace normalisation
pub fn preprocess_text(text: &str) -> String {
    let s = remove_urls(text);
    let s = expand_contractions(&s);
    let s = expand_tokens(&s);
    let s = s.to_lowercase();
    normalize_whitespace(&s)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_number_to_words() {
        assert_eq!(number_to_words(0), "zero");
        assert_eq!(number_to_words(1), "one");
        assert_eq!(number_to_words(15), "fifteen");
        assert_eq!(number_to_words(42), "forty two");
        assert_eq!(number_to_words(100), "one hundred");
        assert_eq!(number_to_words(1000), "one thousand");
        assert_eq!(number_to_words(1_000_000), "one million");
        assert_eq!(number_to_words(1_000_000_000), "one billion");
        assert_eq!(number_to_words(-5), "negative five");
    }

    #[test]
    fn test_currency() {
        assert_eq!(preprocess_text("$100"), "one hundred dollars");
        assert_eq!(preprocess_text("€50.99"), "fifty euros and ninety nine cents");
        assert_eq!(preprocess_text("£20"), "twenty pounds");
        assert_eq!(preprocess_text("¥1000"), "one thousand yen");
    }

    #[test]
    fn test_percentage() {
        assert_eq!(preprocess_text("50%"), "fifty percent");
        assert_eq!(preprocess_text("3.5%"), "three point five percent");
    }

    #[test]
    fn test_units() {
        assert_eq!(preprocess_text("100km"), "one hundred kilometers");
        assert_eq!(preprocess_text("5kg"), "five kilograms");
    }

    #[test]
    fn test_temperature() {
        assert_eq!(preprocess_text("72°F"), "seventy two degrees fahrenheit");
        assert_eq!(preprocess_text("0°C"), "zero degrees celsius");
    }

    #[test]
    fn test_scale() {
        assert_eq!(preprocess_text("7B"), "seven billion");
        assert_eq!(preprocess_text("2.5M"), "two point five million");
        assert_eq!(preprocess_text("1K"), "one thousand");
        assert_eq!(preprocess_text("3T"), "three trillion");
    }

    #[test]
    fn test_fractions() {
        assert_eq!(preprocess_text("1/2"), "one half");
        assert_eq!(preprocess_text("3/4"), "three quarters");
        assert_eq!(preprocess_text("2/3"), "two thirds");
    }

    #[test]
    fn test_scientific() {
        assert_eq!(preprocess_text("1e4"), "one times ten to the four");
        assert_eq!(preprocess_text("1e-4"), "one times ten to the negative four");
    }

    #[test]
    fn test_ip() {
        assert_eq!(preprocess_text("192.168.1.1"), "one nine two dot one six eight dot one dot one");
    }

    #[test]
    fn test_range() {
        assert_eq!(preprocess_text("10-20"), "ten to twenty");
    }

    #[test]
    fn test_contractions() {
        assert_eq!(preprocess_text("don't"), "do not");
        assert_eq!(preprocess_text("won't"), "will not");
        assert_eq!(preprocess_text("I'm"), "i am");
        assert_eq!(preprocess_text("it's"), "it is");
    }

    #[test]
    fn test_url_removal() {
        assert_eq!(preprocess_text("visit https://example.com today"), "visit today");
    }

    #[test]
    fn test_whitespace() {
        assert_eq!(preprocess_text("hello   world"), "hello world");
    }

    #[test]
    fn test_currencies_new() {
        assert_eq!(preprocess_text("$100"), "one hundred dollars");
        assert_eq!(preprocess_text("€50.99"), "fifty euros and ninety nine cents");
        assert_eq!(preprocess_text("£20"), "twenty pounds");
    }

    #[test]
    fn test_units_new() {
        assert_eq!(preprocess_text("5km"), "five kilometers");
        assert_eq!(preprocess_text("120mph"), "one hundred twenty miles per hour");
        assert_eq!(preprocess_text("3kg"), "three kilograms");
    }

    #[test]
    fn test_percentages() {
        assert_eq!(preprocess_text("95%"), "ninety five percent");
    }

    #[test]
    fn test_scales() {
        assert_eq!(preprocess_text("$2.5B"), "two point five billion dollars");
    }

    #[test]
    fn test_fractions_new() {
        assert_eq!(preprocess_text("1/2"), "one half");
        assert_eq!(preprocess_text("3/4"), "three quarters");
    }

    #[test]
    fn test_mixed_sentence() {
        let result = preprocess_text("It costs $100 and is 5km away");
        assert!(result.contains("one hundred dollars"), "got: {result}");
        assert!(result.contains("and is"), "got: {result}");
        assert!(result.contains("five kilometers"), "got: {result}");
    }

    #[test]
    fn test_contractions_new() {
        assert_eq!(preprocess_text("I can't do it"), "i cannot do it");
        assert_eq!(preprocess_text("don't stop"), "do not stop");
    }

    #[test]
    fn test_passthrough() {
        // Plain text should pass through unchanged (just lowercased)
        assert_eq!(preprocess_text("Hello World"), "hello world");
    }
}

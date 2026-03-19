use std::collections::HashMap;
use std::sync::OnceLock;

// Complete 179-symbol table (indices 0-178):
// 0:    '$'  (padding)
// 1-15: punctuation: ; : , . ! ? ¡ ¿ — … " « » \u{201c} \u{201d}
// 16:   ' '  (space)
// 17-42: A-Z uppercase
// 43-68: a-z lowercase
// 69-178: IPA characters

static SYMBOLS: &[&str] = &[
    "$",      // 0
    ";",      // 1
    ":",      // 2
    ",",      // 3
    ".",      // 4
    "!",      // 5
    "?",      // 6
    "¡",      // 7
    "¿",      // 8
    "—",      // 9
    "…",      // 10
    "\"",     // 11
    "«",      // 12
    "»",      // 13
    "\u{201c}", // 14  "
    "\u{201d}", // 15  "
    " ",      // 16
    "A",      // 17
    "B",      // 18
    "C",      // 19
    "D",      // 20
    "E",      // 21
    "F",      // 22
    "G",      // 23
    "H",      // 24
    "I",      // 25
    "J",      // 26
    "K",      // 27
    "L",      // 28
    "M",      // 29
    "N",      // 30
    "O",      // 31
    "P",      // 32
    "Q",      // 33
    "R",      // 34
    "S",      // 35
    "T",      // 36
    "U",      // 37
    "V",      // 38
    "W",      // 39
    "X",      // 40
    "Y",      // 41
    "Z",      // 42
    "a",      // 43
    "b",      // 44
    "c",      // 45
    "d",      // 46
    "e",      // 47
    "f",      // 48
    "g",      // 49
    "h",      // 50
    "i",      // 51
    "j",      // 52
    "k",      // 53
    "l",      // 54
    "m",      // 55
    "n",      // 56
    "o",      // 57
    "p",      // 58
    "q",      // 59
    "r",      // 60
    "s",      // 61
    "t",      // 62
    "u",      // 63
    "v",      // 64
    "w",      // 65
    "x",      // 66
    "y",      // 67
    "z",      // 68
    "ɑ",      // 69
    "ɐ",      // 70
    "ɒ",      // 71
    "æ",      // 72
    "ß",      // 73 (actually IPA β-like use)
    "ɓ",      // 74
    "ʙ",      // 75
    "β",      // 76
    "ɔ",      // 77
    "ɕ",      // 78
    "ç",      // 79
    "ɗ",      // 80
    "ɖ",      // 81
    "ð",      // 82
    "ʤ",      // 83
    "ə",      // 84
    "ɘ",      // 85
    "ɚ",      // 86
    "ɛ",      // 87
    "ɜ",      // 88
    "ɝ",      // 89
    "ɞ",      // 90
    "ɟ",      // 91
    "ʄ",      // 92
    "ɡ",      // 93
    "ɠ",      // 94
    "ʛ",      // 95
    "ɢ",      // 96
    "ħ",      // 97
    "ɦ",      // 98
    "ɧ",      // 99
    "ʜ",      // 100
    "ɪ",      // 101
    "ʝ",      // 102
    "ɭ",      // 103
    "ɬ",      // 104
    "ɫ",      // 105
    "ɮ",      // 106
    "ʟ",      // 107
    "ɱ",      // 108
    "ɯ",      // 109
    "ɰ",      // 110
    "ŋ",      // 111
    "ɳ",      // 112
    "ɲ",      // 113
    "ɴ",      // 114
    "ø",      // 115
    "ɵ",      // 116
    "ɸ",      // 117
    "θ",      // 118
    "œ",      // 119
    "ɶ",      // 120
    "ʘ",      // 121
    "ɹ",      // 122
    "ɺ",      // 123
    "ɾ",      // 124
    "ɻ",      // 125
    "ʀ",      // 126
    "ʁ",      // 127
    "ɽ",      // 128
    "ʂ",      // 129
    "ʃ",      // 130
    "ʈ",      // 131
    "ʧ",      // 132
    "ʉ",      // 133
    "ʊ",      // 134
    "ʋ",      // 135
    "ⱱ",      // 136
    "ʌ",      // 137
    "ɣ",      // 138
    "ɤ",      // 139
    "ʍ",      // 140
    "χ",      // 141
    "ʎ",      // 142
    "ʏ",      // 143
    "ʑ",      // 144
    "ʐ",      // 145
    "ʒ",      // 146
    "ʔ",      // 147
    "ʡ",      // 148
    "ʕ",      // 149
    "ʢ",      // 150
    "ǀ",      // 151
    "ǁ",      // 152
    "ǂ",      // 153
    "ǃ",      // 154
    "ˈ",      // 155
    "ˌ",      // 156
    "ː",      // 157
    "ˑ",      // 158
    "ʼ",      // 159
    "ʴ",      // 160
    "ʵ",      // 161
    "ʶ",      // 162
    "ʷ",      // 163
    "ʸ",      // 164
    "͡",      // 165 (U+0361 combining double inverted breve)
    "͜",      // 166 (U+035C combining double breve below)
    "˞",      // 167
    "↓",      // 168
    "↑",      // 169
    "→",      // 170
    "↗",      // 171
    "↘",      // 172
    "ˠ",      // 173
    "ˤ",      // 174
    "'",      // 175 (apostrophe / left single quote)
    "\u{0329}", // 176 (combining vertical line below)
    "\u{2019}", // 177 (right single quotation mark)
    "ᵻ",      // 178
];

static SYMBOL_TO_ID: OnceLock<HashMap<char, i32>> = OnceLock::new();

fn get_symbol_to_id() -> &'static HashMap<char, i32> {
    SYMBOL_TO_ID.get_or_init(|| {
        let mut map = HashMap::new();
        for (idx, sym) in SYMBOLS.iter().enumerate() {
            // Each entry is a single char (or combining char sequence stored as str).
            // We key by the first char of the symbol string; for multi-char entries
            // like the combining char (index 176), the str IS the char sequence.
            // For single-char symbols we insert the char directly.
            let mut chars = sym.chars();
            if let Some(ch) = chars.next() {
                if chars.next().is_none() {
                    // Single char — straightforward insert.
                    // For duplicate entries (e.g. ħ at 97 and 100), last write wins.
                    map.insert(ch, idx as i32);
                } else {
                    // Multi-char entry (combining sequences): insert the lead char
                    // keyed to this index so lookup works char-by-char.
                    // This is a best-effort; combining sequences need special handling.
                    map.insert(ch, idx as i32);
                }
            }
        }
        map
    })
}

pub fn map_phonemes_to_ids(ipa: &str) -> Vec<i32> {
    let map = get_symbol_to_id();
    let mut ids = vec![0i32]; // start token
    for ch in ipa.chars() {
        if let Some(&id) = map.get(&ch) {
            ids.push(id);
        }
        // silently skip unknown chars (matches Python behavior)
    }
    ids.push(10); // end token (…)
    ids.push(0);  // end padding
    ids
}

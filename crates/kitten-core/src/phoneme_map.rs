use std::collections::HashMap;
use std::sync::OnceLock;

// Complete 179-symbol table (indices 0-178):
// 0:    '$'  (padding)
// 1-15: punctuation: ; : , . ! ? ¡ ¿ — … " « » \u{201c} \u{201d}
// 16:   ' '  (space)
// 17-42: A-Z uppercase
// 43-68: a-z lowercase
// 69-178: IPA characters
//
// Python reference (kitten_reference_generate.py):
//   IPA_CHARS = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʓʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
//   IPA starts at index 69.

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
    // IPA characters matching Python IPA_CHARS exactly (offset 69):
    "ɑ",      // 69
    "ɐ",      // 70
    "ɒ",      // 71
    "æ",      // 72
    "ɓ",      // 73
    "ʙ",      // 74
    "β",      // 75
    "ɔ",      // 76
    "ɕ",      // 77
    "ç",      // 78
    "ɗ",      // 79
    "ɖ",      // 80
    "ð",      // 81
    "ʤ",      // 82
    "ə",      // 83
    "ɘ",      // 84
    "ɚ",      // 85
    "ɛ",      // 86
    "ɜ",      // 87
    "ɝ",      // 88
    "ɞ",      // 89
    "ɟ",      // 90
    "ʄ",      // 91
    "ɡ",      // 92
    "ɠ",      // 93
    "ɢ",      // 94
    "ʛ",      // 95
    "ɦ",      // 96
    "ɧ",      // 97
    "ħ",      // 98
    "ɥ",      // 99
    "ʜ",      // 100
    "ɨ",      // 101
    "ɪ",      // 102
    "ʝ",      // 103
    "ɭ",      // 104
    "ɬ",      // 105
    "ɫ",      // 106
    "ɮ",      // 107
    "ʟ",      // 108
    "ɱ",      // 109
    "ɯ",      // 110
    "ɰ",      // 111
    "ŋ",      // 112
    "ɳ",      // 113
    "ɲ",      // 114
    "ɴ",      // 115
    "ø",      // 116
    "ɵ",      // 117
    "ɸ",      // 118
    "θ",      // 119
    "œ",      // 120
    "ɶ",      // 121
    "ʘ",      // 122
    "ɹ",      // 123
    "ɺ",      // 124
    "ɾ",      // 125
    "ɻ",      // 126
    "ʀ",      // 127
    "ʁ",      // 128
    "ɽ",      // 129
    "ʂ",      // 130
    "ʃ",      // 131
    "ʈ",      // 132
    "ʧ",      // 133
    "ʉ",      // 134
    "ʊ",      // 135
    "ʋ",      // 136
    "ⱱ",      // 137
    "ʌ",      // 138
    "ɣ",      // 139
    "ɤ",      // 140
    "ʍ",      // 141
    "χ",      // 142
    "ʎ",      // 143
    "ʏ",      // 144
    "ʑ",      // 145
    "ʐ",      // 146
    "ʒ",      // 147
    "ʓ",      // 148
    "ʔ",      // 149
    "ʡ",      // 150
    "ʕ",      // 151
    "ʢ",      // 152
    "ǀ",      // 153
    "ǁ",      // 154
    "ǂ",      // 155
    "ǃ",      // 156
    "ˈ",      // 157
    "ˌ",      // 158
    "ː",      // 159
    "ˑ",      // 160
    "ʼ",      // 161
    "ʴ",      // 162
    "ʰ",      // 163
    "ʱ",      // 164
    "ʲ",      // 165
    "ʷ",      // 166
    "ˠ",      // 167
    "ˤ",      // 168
    "˞",      // 169
    "↓",      // 170
    "↑",      // 171
    "→",      // 172
    "↗",      // 173
    "↘",      // 174
    "'",      // 175
    "\u{0329}", // 176 (combining vertical line below — ̩)
    "\u{2019}", // 177 (right single quotation mark — ')
    "ᵻ",      // 178
];

static SYMBOL_TO_ID: OnceLock<HashMap<char, i32>> = OnceLock::new();

fn get_symbol_to_id() -> &'static HashMap<char, i32> {
    SYMBOL_TO_ID.get_or_init(|| {
        let mut map = HashMap::new();
        for (idx, sym) in SYMBOLS.iter().enumerate() {
            let mut chars = sym.chars();
            if let Some(ch) = chars.next() {
                map.insert(ch, idx as i32);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hello_world_phoneme_ids() {
        // "həlˈəʊ wˈɜːld" → ONNX reference:
        // [0, 50, 83, 54, 157, 83, 135, 16, 65, 157, 87, 159, 54, 46, 10, 0]
        let result = map_phonemes_to_ids("həlˈəʊ wˈɜːld");
        assert_eq!(
            result,
            vec![0, 50, 83, 54, 157, 83, 135, 16, 65, 157, 87, 159, 54, 46, 10, 0],
            "Phoneme IDs don't match ONNX reference"
        );
    }
}

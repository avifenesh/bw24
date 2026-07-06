//! Minimal recursive-descent JSON parser for HF tokenizer sidecar files
//! (tokenizer.json / tokenizer_config.json / generation_config.json).
//!
//! bw24-tokenizer stays serde-free (crate policy, same as bw24-gguf's hand
//! parser in safetensors.rs). This is a full-value parser (objects, arrays,
//! strings with \uXXXX + surrogate pairs, numbers, bools, null) — the
//! tokenizer.json vocab is ~1e5 entries so parsing is byte-based, no regex.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Value>),
    Obj(HashMap<String, Value>),
}

impl Value {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Num(n) if *n >= 0.0 && n.fract() == 0.0 => Some(*n as u64),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }
    pub fn as_arr(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(a) => Some(a),
            _ => None,
        }
    }
    pub fn as_obj(&self) -> Option<&HashMap<String, Value>> {
        match self {
            Value::Obj(o) => Some(o),
            _ => None,
        }
    }
    /// obj["key"] convenience (None on non-objects / missing keys).
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.as_obj().and_then(|o| o.get(key))
    }
}

pub fn parse(text: &str) -> Result<Value, String> {
    let mut p = Parser { b: text.as_bytes(), i: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.b.len() {
        return Err(format!("json: trailing bytes at {}", p.i));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn peek(&self) -> Result<u8, String> {
        self.b.get(self.i).copied().ok_or_else(|| "json: unexpected EOF".to_string())
    }

    fn expect(&mut self, c: u8) -> Result<(), String> {
        if self.peek()? != c {
            return Err(format!("json: expected '{}' at byte {}, got '{}'",
                               c as char, self.i, self.b[self.i] as char));
        }
        self.i += 1;
        Ok(())
    }

    fn value(&mut self) -> Result<Value, String> {
        match self.peek()? {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => Ok(Value::Str(self.string()?)),
            b't' => self.lit(b"true", Value::Bool(true)),
            b'f' => self.lit(b"false", Value::Bool(false)),
            b'n' => self.lit(b"null", Value::Null),
            _ => self.number(),
        }
    }

    fn lit(&mut self, word: &[u8], v: Value) -> Result<Value, String> {
        if self.b.len() - self.i >= word.len() && &self.b[self.i..self.i + word.len()] == word {
            self.i += word.len();
            Ok(v)
        } else {
            Err(format!("json: bad literal at byte {}", self.i))
        }
    }

    fn number(&mut self) -> Result<Value, String> {
        let start = self.i;
        while self.i < self.b.len()
            && matches!(self.b[self.i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
        {
            self.i += 1;
        }
        if self.i == start {
            return Err(format!("json: expected value at byte {}", start));
        }
        std::str::from_utf8(&self.b[start..self.i])
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .map(Value::Num)
            .ok_or_else(|| format!("json: bad number at byte {start}"))
    }

    fn hex4(&mut self) -> Result<u32, String> {
        if self.i + 4 > self.b.len() {
            return Err("json: truncated \\u escape".into());
        }
        let s = std::str::from_utf8(&self.b[self.i..self.i + 4])
            .map_err(|_| "json: bad \\u escape".to_string())?;
        let v = u32::from_str_radix(s, 16).map_err(|_| "json: bad \\u escape".to_string())?;
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let c = self.peek()?;
            match c {
                b'"' => {
                    self.i += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.i += 1;
                    let e = self.peek()?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                // surrogate pair: require \uXXXX low surrogate
                                if self.i + 2 > self.b.len()
                                    || self.b[self.i] != b'\\'
                                    || self.b[self.i + 1] != b'u'
                                {
                                    return Err("json: lone high surrogate".into());
                                }
                                self.i += 2;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err("json: bad low surrogate".into());
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                hi
                            };
                            out.push(char::from_u32(cp)
                                .ok_or_else(|| "json: invalid codepoint".to_string())?);
                        }
                        _ => return Err(format!("json: bad escape at byte {}", self.i - 1)),
                    }
                }
                _ => {
                    // consume one UTF-8 sequence verbatim (fast path: run of plain bytes)
                    let start = self.i;
                    while self.i < self.b.len()
                        && self.b[self.i] != b'"'
                        && self.b[self.i] != b'\\'
                    {
                        self.i += 1;
                    }
                    out.push_str(
                        std::str::from_utf8(&self.b[start..self.i])
                            .map_err(|_| "json: invalid utf-8 in string".to_string())?,
                    );
                }
            }
        }
    }

    fn array(&mut self) -> Result<Value, String> {
        self.expect(b'[')?;
        let mut out = Vec::new();
        self.ws();
        if self.peek()? == b']' {
            self.i += 1;
            return Ok(Value::Arr(out));
        }
        loop {
            self.ws();
            out.push(self.value()?);
            self.ws();
            match self.peek()? {
                b',' => self.i += 1,
                b']' => {
                    self.i += 1;
                    return Ok(Value::Arr(out));
                }
                c => return Err(format!("json: expected ',' or ']' at byte {}, got '{}'",
                                        self.i, c as char)),
            }
        }
    }

    fn object(&mut self) -> Result<Value, String> {
        self.expect(b'{')?;
        let mut out = HashMap::new();
        self.ws();
        if self.peek()? == b'}' {
            self.i += 1;
            return Ok(Value::Obj(out));
        }
        loop {
            self.ws();
            let k = self.string()?;
            self.ws();
            self.expect(b':')?;
            self.ws();
            let v = self.value()?;
            out.insert(k, v);
            self.ws();
            match self.peek()? {
                b',' => self.i += 1,
                b'}' => {
                    self.i += 1;
                    return Ok(Value::Obj(out));
                }
                c => return Err(format!("json: expected ',' or '}}' at byte {}, got '{}'",
                                        self.i, c as char)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalars_and_nesting() {
        let v = parse(r#"{"a": [1, -2.5, "xĠy", true, null], "b": {"c": "😀"}}"#)
            .unwrap();
        assert_eq!(v.get("a").unwrap().as_arr().unwrap()[0].as_u64(), Some(1));
        assert_eq!(v.get("a").unwrap().as_arr().unwrap()[2].as_str(), Some("x\u{120}y"));
        assert_eq!(v.get("b").unwrap().get("c").unwrap().as_str(), Some("\u{1F600}"));
    }
}

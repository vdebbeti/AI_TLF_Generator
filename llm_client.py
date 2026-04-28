import base64


PROVIDER_MODELS: dict[str, list[str]] = {
    "OpenAI": ["gpt-4o", "gpt-4o-mini"],
    "Google Gemini": ["gemini-2.0-flash", "gemini-1.5-pro"],
    "Anthropic Claude": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
}


def call_llm(
    system: str,
    user: str,
    provider: str,
    model: str,
    api_key: str,
    image_bytes: bytes | None = None,
    image_mime: str = "image/png",
    max_tokens: int = 3000,
    temperature: float = 0.2,
    json_mode: bool = False,
) -> str:
    if provider == "OpenAI":
        return _call_openai(
            system, user, model, api_key, image_bytes, image_mime, max_tokens, temperature, json_mode
        )
    if provider == "Google Gemini":
        return _call_gemini(system, user, model, api_key, image_bytes, image_mime, max_tokens, temperature)
    if provider == "Anthropic Claude":
        return _call_claude(system, user, model, api_key, image_bytes, image_mime, max_tokens, temperature)
    raise ValueError(f"Unknown provider: {provider}")


def _call_openai(system, user, model, api_key, image_bytes, image_mime, max_tokens, temperature, json_mode):
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    if image_bytes:
        b64 = base64.b64encode(image_bytes).decode()
        user_content: list = [
            {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{b64}", "detail": "high"}},
            {"type": "text", "text": user},
        ]
    else:
        user_content = user

    kwargs = {}
    if json_mode and not image_bytes:
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )
    return (resp.choices[0].message.content or "").strip()


def _call_gemini(system, user, model, api_key, image_bytes, image_mime, max_tokens, temperature):
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    generation_config = genai.GenerationConfig(max_output_tokens=max_tokens, temperature=temperature)
    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system,
        generation_config=generation_config,
    )
    parts: list = []
    if image_bytes:
        b64 = base64.b64encode(image_bytes).decode()
        parts.append({"inline_data": {"mime_type": image_mime, "data": b64}})
    parts.append(user)
    resp = gemini_model.generate_content(parts)
    return (resp.text or "").strip()


def _call_claude(system, user, model, api_key, image_bytes, image_mime, max_tokens, temperature):
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    content: list = []
    if image_bytes:
        b64 = base64.b64encode(image_bytes).decode()
        content.append(
            {"type": "image", "source": {"type": "base64", "media_type": image_mime, "data": b64}}
        )
    content.append({"type": "text", "text": user})

    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": content}],
    )
    return (resp.content[0].text if resp.content else "").strip()


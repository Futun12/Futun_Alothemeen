"""Lab 2 starter notebook-as-script.
Complete the marked sections, verify numerical equivalence, inspect parameter
accounting, causal masking, attention heads and pad-attention leakage.
"""

import math
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from bayan.attention import attention, MultiHeadAttention


def main():
    # Make random values reproducible.
    torch.manual_seed(42)

    # ---------------------------------------------------------
    # 1. Create small Q, K, V tensors
    # ---------------------------------------------------------
    q = torch.randn(1, 3, 4)
    k = torch.randn(1, 3, 4)
    v = torch.randn(1, 3, 4)

    print("Q shape:", q.shape)
    print("K shape:", k.shape)
    print("V shape:", v.shape)

    # ---------------------------------------------------------
    # 2. Verify numerical equivalence
    # ---------------------------------------------------------
    our_output = attention(q, k, v)

    torch_output = F.scaled_dot_product_attention(
        q,
        k,
        v
    )

    print("\nOur attention output:")
    print(our_output)

    print("\nPyTorch attention output:")
    print(torch_output)

    equivalent = torch.allclose(
        our_output,
        torch_output,
        atol=1e-6
    )

    print("\nNumerically equivalent:")
    print(equivalent)

    # ---------------------------------------------------------
    # 3. Inspect attention weight matrix
    # ---------------------------------------------------------
    d_k = k.size(-1)

    scores = torch.matmul(
        q,
        k.transpose(-2, -1)
    )

    scores = scores / math.sqrt(d_k)

    weights = torch.softmax(
        scores,
        dim=-1
    )

    print("\nAttention scores:")
    print(scores)

    print("\nAttention weight matrix:")
    print(weights)

    print("\nAttention row sums:")
    print(weights.sum(dim=-1))

    # ---------------------------------------------------------
    # 4. Exercise Multi-Head Attention
    # ---------------------------------------------------------
    d_model = 8
    num_heads = 2

    mha = MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads
    )

    x = torch.randn(
        1,
        4,
        d_model
    )

    mha_output = mha(x)

    print("\nMulti-Head Attention input shape:")
    print(x.shape)

    print("\nNumber of attention heads:")
    print(num_heads)

    print("\nDimensions per head:")
    print(d_model // num_heads)

    print("\nMulti-Head Attention output shape:")
    print(mha_output.shape)

    # ---------------------------------------------------------
    # 5. Parameter accounting
    # ---------------------------------------------------------
    parameter_count = sum(
        parameter.numel()
        for parameter in mha.parameters()
    )

    print("\nMulti-Head Attention parameter count:")
    print(parameter_count)

    # ---------------------------------------------------------
    # 6. Verify causal masking
    # ---------------------------------------------------------

    # Create a lower-triangular mask.
    # Token position i can only attend to positions <= i.
    causal_mask = torch.tril(
        torch.ones(3, 3)
    )

    print("\nCausal mask:")
    print(causal_mask)

    # Apply the causal mask using our attention function.
    causal_output = attention(
        q,
        k,
        v,
        mask=causal_mask
    )

    # Apply the mask to the scores.
    # Future positions become -infinity.
    causal_scores = scores.masked_fill(
        causal_mask == 0,
        float("-inf")
    )

    # Convert masked scores into attention weights.
    causal_weights = torch.softmax(
        causal_scores,
        dim=-1
    )

    print("\nAttention weights with causal mask:")
    print(causal_weights)

    print("\nOutput with causal mask:")
    print(causal_output)

    # Find all positions above the diagonal.
    # These are future positions that must not receive attention.
    future_positions = torch.triu(
        torch.ones(
            3,
            3,
            dtype=torch.bool
        ),
        diagonal=1
    )

    # Get attention values given to future positions.
    future_attention = causal_weights[0][
        future_positions
    ]

    print("\nAttention given to future positions:")
    print(future_attention)

    # Verify that all future attention values are zero.
    print("\nCausal masking works:")
    print(
        torch.allclose(
            future_attention,
            torch.zeros_like(future_attention)
        )
    )

    print("\nModel family:")
    print("Decoder-style causal attention")

    # ---------------------------------------------------------
    # 7. Verify padding mask
    # ---------------------------------------------------------
    pad_mask = torch.tensor(
        [[[1, 1, 0]]]
    )

    print("\nPadding mask:")
    print(pad_mask)

    pad_output = attention(
        q,
        k,
        v,
        mask=pad_mask
    )

    pad_scores = scores.masked_fill(
        pad_mask == 0,
        float("-inf")
    )

    pad_weights = torch.softmax(
        pad_scores,
        dim=-1
    )

    print("\nAttention weights with padding mask:")
    print(pad_weights)

    print("\nAttention given to padded position:")
    print(pad_weights[..., 2])

    print("\nOutput with padding mask:")
    print(pad_output)

    # Verify that the padded position receives zero attention.
    pad_attention = pad_weights[..., 2]

    print("\nPadding mask prevents attention leakage:")
    print(
        torch.allclose(
            pad_attention,
            torch.zeros_like(pad_attention)
        )
    )

    # =========================================================
    # LAB 2 STEP 5
    # Attention-map diagnostics on Bayan examples
    # =========================================================

    print("\n" + "=" * 70)
    print("LAB 2 STEP 5: ATTENTION-MAP DIAGNOSTICS")
    print("=" * 70)

    # ---------------------------------------------------------
    # 8. Load tokenizer and pretrained model
    # ---------------------------------------------------------
    checkpoint = "bert-base-multilingual-cased"

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint
    )

    model = AutoModel.from_pretrained(
        checkpoint,
        output_attentions=True,
        attn_implementation="eager"
    )

    model.eval()

    # ---------------------------------------------------------
    # 9. Bayan example sentences
    # ---------------------------------------------------------
    examples = [
        "الخدمة ممتازة ولكن الانتظار طويل.",
        "The service was good."
    ]

    # Tokenize both examples together.
    # padding=True makes the shorter example contain [PAD].
    encoded = tokenizer(
        examples,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    print("\nInput IDs:")
    print(input_ids)

    print("\nAttention mask:")
    print(attention_mask)

    # Convert token IDs back into readable tokens.
    token_lists = []

    for ids in input_ids:
        token_list = tokenizer.convert_ids_to_tokens(
            ids.tolist()
        )

        token_lists.append(token_list)

    print("\nTokenized Bayan examples:")

    for example_index, token_list in enumerate(token_lists):
        print(f"\nExample {example_index + 1}:")
        print(token_list)

    # ---------------------------------------------------------
    # 10. Run model WITH correct attention mask
    # ---------------------------------------------------------
    with torch.no_grad():
        outputs_with_mask = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )

    attentions_with_mask = outputs_with_mask.attentions

    # We inspect the final Transformer layer.
    last_layer_with_mask = attentions_with_mask[-1]

    print("\nLast-layer attention shape:")
    print(last_layer_with_mask.shape)

    # Shape is:
    # [batch, number_of_heads, sequence_length, sequence_length]

    # ---------------------------------------------------------
    # 11. Run model WITHOUT attention mask
    # ---------------------------------------------------------
    # This intentionally demonstrates incorrect behaviour.
    with torch.no_grad():
        outputs_without_mask = model(
            input_ids=input_ids,
            output_attentions=True
        )

    attentions_without_mask = outputs_without_mask.attentions

    last_layer_without_mask = attentions_without_mask[-1]

    # ---------------------------------------------------------
    # 12. Inspect attention heads
    # ---------------------------------------------------------
    number_of_heads = last_layer_with_mask.size(1)

    print("\nNumber of pretrained attention heads:")
    print(number_of_heads)

    for example_index in range(len(examples)):

        print("\n" + "-" * 70)
        print(f"BAYAN EXAMPLE {example_index + 1}")
        print("-" * 70)

        token_list = token_lists[example_index]

        # Number of real tokens, excluding padding.
        valid_length = int(
            attention_mask[
                example_index
            ].sum().item()
        )

        print("\nTokens:")
        print(token_list)

        print("\nNumber of real tokens:")
        print(valid_length)

        # Find [SEP].
        sep_positions = [
            index
            for index, token in enumerate(token_list)
            if token == "[SEP]"
        ]

        # Find [PAD] positions.
        pad_positions = [
            index
            for index, mask_value
            in enumerate(attention_mask[example_index])
            if mask_value.item() == 0
        ]

        print("\n[SEP] positions:")
        print(sep_positions)

        print("\n[PAD] positions:")
        print(pad_positions)

        # -----------------------------------------------------
        # Inspect every attention head
        # -----------------------------------------------------
        for head_index in range(number_of_heads):

            attention_with_mask = last_layer_with_mask[
                example_index,
                head_index
            ]

            attention_without_mask = last_layer_without_mask[
                example_index,
                head_index
            ]

            # -------------------------------------------------
            # A. Adjacency-looking behaviour
            # -------------------------------------------------
            adjacency_mass = 0.0
            adjacency_count = 0

            for token_index in range(valid_length):

                # Attention to previous token.
                if token_index - 1 >= 0:
                    adjacency_mass += attention_with_mask[
                        token_index,
                        token_index - 1
                    ].item()

                    adjacency_count += 1

                # Attention to next token.
                if token_index + 1 < valid_length:
                    adjacency_mass += attention_with_mask[
                        token_index,
                        token_index + 1
                    ].item()

                    adjacency_count += 1

            if adjacency_count > 0:
                average_adjacency = (
                    adjacency_mass /
                    adjacency_count
                )
            else:
                average_adjacency = 0.0

            # -------------------------------------------------
            # B. [SEP] sink behaviour
            # -------------------------------------------------
            sep_mass = 0.0

            if len(sep_positions) > 0:
                sep_index = sep_positions[0]

                sep_mass = attention_with_mask[
                    :valid_length,
                    sep_index
                ].mean().item()

            # -------------------------------------------------
            # C. PAD mass
            # -------------------------------------------------
            pad_mass_with_mask = 0.0
            pad_mass_without_mask = 0.0

            if len(pad_positions) > 0:

                # Total attention paid to all PAD positions
                # by each real query token.
                pad_mass_with_mask = attention_with_mask[
                    :valid_length,
                    pad_positions
                ].sum(
                    dim=-1
                ).mean().item()

                pad_mass_without_mask = attention_without_mask[
                    :valid_length,
                    pad_positions
                ].sum(
                    dim=-1
                ).mean().item()

            print(
                f"\nHead {head_index}:"
            )

            print(
                "  Average adjacent-token attention:",
                round(average_adjacency, 6)
            )

            print(
                "  Average [SEP] attention:",
                round(sep_mass, 6)
            )

            if len(pad_positions) > 0:
                print(
                    "  PAD mass WITH correct mask:",
                    round(
                        pad_mass_with_mask,
                        6
                    )
                )

                print(
                    "  PAD mass WITHOUT mask:",
                    round(
                        pad_mass_without_mask,
                        6
                    )
                )

    # ---------------------------------------------------------
    # 13. Automatically find strongest adjacency-looking head
    # ---------------------------------------------------------
    example_index = 0

    valid_length = int(
        attention_mask[
            example_index
        ].sum().item()
    )

    best_adjacency_head = None
    best_adjacency_score = -1.0

    for head_index in range(number_of_heads):

        head_attention = last_layer_with_mask[
            example_index,
            head_index
        ]

        adjacency_mass = 0.0
        adjacency_count = 0

        for token_index in range(valid_length):

            if token_index - 1 >= 0:
                adjacency_mass += head_attention[
                    token_index,
                    token_index - 1
                ].item()

                adjacency_count += 1

            if token_index + 1 < valid_length:
                adjacency_mass += head_attention[
                    token_index,
                    token_index + 1
                ].item()

                adjacency_count += 1

        if adjacency_count > 0:
            adjacency_score = (
                adjacency_mass /
                adjacency_count
            )
        else:
            adjacency_score = 0.0

        if adjacency_score > best_adjacency_score:
            best_adjacency_score = adjacency_score
            best_adjacency_head = head_index

    print("\n" + "=" * 70)
    print("ADJACENCY DIAGNOSTIC")
    print("=" * 70)

    print("\nStrongest adjacency-looking head:")
    print(best_adjacency_head)

    print("\nAdjacency score:")
    print(round(best_adjacency_score, 6))

    # ---------------------------------------------------------
    # 14. Automatically find strongest [SEP] sink head
    # ---------------------------------------------------------
    sep_positions = [
        index
        for index, token
        in enumerate(token_lists[example_index])
        if token == "[SEP]"
    ]

    best_sep_head = None
    best_sep_score = -1.0

    if len(sep_positions) > 0:

        sep_index = sep_positions[0]

        for head_index in range(number_of_heads):

            head_attention = last_layer_with_mask[
                example_index,
                head_index
            ]

            sep_score = head_attention[
                :valid_length,
                sep_index
            ].mean().item()

            if sep_score > best_sep_score:
                best_sep_score = sep_score
                best_sep_head = head_index

    print("\n" + "=" * 70)
    print("[SEP] SINK DIAGNOSTIC")
    print("=" * 70)

    print("\nStrongest [SEP] sink head:")
    print(best_sep_head)

    print("\nAverage attention to [SEP]:")
    print(round(best_sep_score, 6))

    # ---------------------------------------------------------
    # 15. Overall PAD leak comparison
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("PAD LEAK DIAGNOSTIC")
    print("=" * 70)

    for example_index in range(len(examples)):

        pad_positions = [
            index
            for index, mask_value
            in enumerate(attention_mask[example_index])
            if mask_value.item() == 0
        ]

        if len(pad_positions) == 0:
            print(
                f"\nExample {example_index + 1}: "
                "No [PAD] tokens."
            )

            continue

        valid_length = int(
            attention_mask[
                example_index
            ].sum().item()
        )

        # Average over all heads and all real query tokens.
        pad_mass_with_mask = last_layer_with_mask[
            example_index,
            :,
            :valid_length,
            pad_positions
        ].sum(
            dim=-1
        ).mean().item()

        pad_mass_without_mask = last_layer_without_mask[
            example_index,
            :,
            :valid_length,
            pad_positions
        ].sum(
            dim=-1
        ).mean().item()

        print(
            f"\nExample {example_index + 1}"
        )

        print(
            "Average PAD mass WITH correct mask:"
        )
        print(
            round(
                pad_mass_with_mask,
                6
            )
        )

        print(
            "Average PAD mass WITHOUT mask:"
        )
        print(
            round(
                pad_mass_without_mask,
                6
            )
        )

        print(
            "PAD leakage prevented by mask:"
        )

        print(
            pad_mass_with_mask
            < pad_mass_without_mask
        )

    # ---------------------------------------------------------
    # 16. Print attention matrix for strongest adjacency head
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("ATTENTION MATRIX FOR STRONGEST ADJACENCY HEAD")
    print("=" * 70)

    print("\nTokens:")
    print(
        token_lists[0][
            :valid_length
        ]
    )

    print("\nAttention matrix:")
    print(
        last_layer_with_mask[
            0,
            best_adjacency_head,
            :valid_length,
            :valid_length
        ]
    )


if __name__ == "__main__":
    main()
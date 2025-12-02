<script lang="ts">
    interface Props {
        fromHub: boolean;
    }

    let { fromHub }: Props = $props();
</script>

<div class="p-4 bg-blue-50 border border-blue-200 rounded-lg">
    <h4 class="text-sm font-semibold text-blue-900 mb-2">
        📋 Vision Dataset Format
    </h4>
    {#if fromHub}
        <p class="text-sm text-blue-800 mb-2">
            HuggingFace datasets should use OpenAI messages format with base64
            images:
        </p>
        <pre class="text-xs bg-blue-100 p-2 rounded overflow-x-auto"><code
                >{`{"messages": [{"role": "user", "content": [{"type": "image", "image": "data:image/jpeg;base64,..."}, {"type": "text", "text": "What is shown?"}]}]}`}</code
            ></pre>
        <p class="text-xs text-blue-700 mt-2">
            <strong>Example:</strong>
            <code>Barth371/train_pop_valet_no_wrong_doc</code>
        </p>
    {:else}
        <p class="text-sm text-blue-800 mb-2">
            Your dataset should be in JSONL format with:
        </p>
        <pre class="text-xs bg-blue-100 p-2 rounded overflow-x-auto"><code
                >{`{"text": "What is in this image?", "image": "/path/to/image.jpg", "response": "A cat sitting on a table"}`}</code
            ></pre>
        <p class="text-xs text-blue-700 mt-2">
            <strong>Tip:</strong> Use
            <code>model-garden create-vision-dataset</code> CLI to generate sample
            data
        </p>
    {/if}
</div>

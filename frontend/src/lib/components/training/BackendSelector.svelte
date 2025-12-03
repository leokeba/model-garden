<script lang="ts">
    import type { TrainingBackend } from "$lib/api/client";

    interface Props {
        backend: string;
        modelType: "text" | "vision";
        backends: TrainingBackend[];
        loading: boolean;
        onBackendChange: (value: string) => void;
    }

    let { backend, modelType, backends, loading, onBackendChange }: Props =
        $props();

    // Filter backends based on model type
    let availableBackends = $derived(
        backends.filter((b) =>
            modelType === "vision" ? b.supports_vision : b.supports_text,
        ),
    );

    // Get selected backend info
    let selectedBackend = $derived(backends.find((b) => b.name === backend));
</script>

<div>
    <h3 class="text-lg font-semibold text-gray-900 mb-4">Training Backend</h3>

    <div>
        <label
            for="backend"
            class="block text-sm font-medium text-gray-700 mb-2"
        >
            Backend
        </label>
        {#if loading}
            <div
                class="w-full px-3 py-2 border border-gray-200 rounded-lg bg-gray-50 text-gray-500"
            >
                Loading backends...
            </div>
        {:else}
            <select
                id="backend"
                value={backend}
                onchange={(e) => onBackendChange(e.currentTarget.value)}
                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
                {#each availableBackends as b}
                    <option value={b.name}>{b.name}</option>
                {/each}
            </select>
        {/if}

        {#if selectedBackend}
            <div class="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                <p class="text-sm text-blue-800">
                    <strong class="capitalize">{selectedBackend.name}:</strong>
                    {selectedBackend.description}
                </p>
                <div class="mt-2 flex gap-2">
                    {#if selectedBackend.supports_text}
                        <span
                            class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800"
                        >
                            Text Models
                        </span>
                    {/if}
                    {#if selectedBackend.supports_vision}
                        <span
                            class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-800"
                        >
                            Vision Models
                        </span>
                    {/if}
                </div>
            </div>
        {/if}
    </div>
</div>

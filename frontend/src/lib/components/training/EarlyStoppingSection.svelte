<script lang="ts">
    interface Props {
        earlyStoppingEnabled: boolean;
        earlyStoppingPatience: number;
        earlyStoppingThreshold: number;
        onEarlyStoppingEnabledChange: (value: boolean) => void;
        onEarlyStoppingPatienceChange: (value: number) => void;
        onEarlyStoppingThresholdChange: (value: number) => void;
    }

    let {
        earlyStoppingEnabled,
        earlyStoppingPatience,
        earlyStoppingThreshold,
        onEarlyStoppingEnabledChange,
        onEarlyStoppingPatienceChange,
        onEarlyStoppingThresholdChange,
    }: Props = $props();
</script>

<div class="mb-6">
    <h4 class="text-md font-medium text-gray-800 mb-3 flex items-center gap-2">
        ⏸️ Automatic Early Stopping
    </h4>

    <div class="p-4 bg-blue-50 border border-blue-200 rounded-lg mb-4">
        <p class="text-sm text-blue-800 mb-2">
            <strong>Automatic Early Stopping:</strong> Stops training when validation
            loss stops improving, preventing overfitting and saving compute time.
        </p>
        <p class="text-xs text-blue-700">
            This is different from the manual "Stop Early" button on the
            training page. This monitors validation metrics and stops
            automatically.
        </p>
    </div>

    <div class="space-y-4">
        <div>
            <div class="flex items-center">
                <input
                    type="checkbox"
                    id="early_stopping_enabled"
                    checked={earlyStoppingEnabled}
                    onchange={(e) =>
                        onEarlyStoppingEnabledChange(e.currentTarget.checked)}
                    class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                />
                <label
                    for="early_stopping_enabled"
                    class="ml-2 block text-sm font-medium text-gray-700"
                >
                    Enable Automatic Early Stopping
                </label>
            </div>
            <p class="text-xs text-gray-500 mt-1 ml-6">
                Monitor validation loss and stop when it stops improving
            </p>
        </div>

        {#if earlyStoppingEnabled}
            <div
                class="ml-6 space-y-4 p-4 bg-white border border-gray-200 rounded-lg"
            >
                <div>
                    <label
                        for="early_stopping_patience"
                        class="block text-sm font-medium text-gray-700 mb-1"
                    >
                        Patience (evaluations)
                    </label>
                    <input
                        type="number"
                        id="early_stopping_patience"
                        value={earlyStoppingPatience}
                        oninput={(e) =>
                            onEarlyStoppingPatienceChange(
                                parseInt(e.currentTarget.value),
                            )}
                        min="1"
                        max="20"
                        class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    />
                    <p class="text-xs text-gray-500 mt-1">
                        Number of evaluations with no improvement before
                        stopping (3-5 typical)
                    </p>
                </div>

                <div>
                    <label
                        for="early_stopping_threshold"
                        class="block text-sm font-medium text-gray-700 mb-1"
                    >
                        Improvement Threshold
                    </label>
                    <input
                        type="number"
                        id="early_stopping_threshold"
                        value={earlyStoppingThreshold}
                        oninput={(e) =>
                            onEarlyStoppingThresholdChange(
                                parseFloat(e.currentTarget.value),
                            )}
                        min="0"
                        step="0.0001"
                        class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    />
                    <p class="text-xs text-gray-500 mt-1">
                        Minimum change to qualify as improvement (0.0001 =
                        0.01%, smaller = more sensitive)
                    </p>
                </div>

                <div class="p-3 bg-green-50 border border-green-200 rounded-lg">
                    <p class="text-xs text-green-800">
                        <strong>💡 Example:</strong> With patience=3 and threshold=0.0001,
                        training stops if validation loss doesn't improve by at least
                        0.01% for 3 consecutive evaluations.
                    </p>
                </div>
            </div>
        {/if}
    </div>
</div>

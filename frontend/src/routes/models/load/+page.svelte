<script lang="ts">
  import { goto } from "$app/navigation";
  import { page } from "$app/stores";
  import { onMount } from "svelte";

  // Redirect to inference page, preserving query params
  onMount(() => {
    const modelParam = $page.url.searchParams.get("model");
    const hfModelParam = $page.url.searchParams.get("hf_model");

    let redirectUrl = "/inference";
    const params = new URLSearchParams();

    if (modelParam) params.set("model", modelParam);
    if (hfModelParam) params.set("hf_model", hfModelParam);

    if (params.toString()) {
      redirectUrl += "?" + params.toString();
    }

    goto(redirectUrl, { replaceState: true });
  });
</script>

<svelte:head>
  <title>Redirecting... - Model Garden</title>
</svelte:head>

<div class="min-h-screen bg-gray-50 flex items-center justify-center">
  <div class="text-center">
    <div
      class="inline-block w-8 h-8 border-4 border-primary-600 border-t-transparent rounded-full animate-spin"
    ></div>
    <p class="mt-4 text-gray-600">Redirecting to Inference...</p>
  </div>
</div>

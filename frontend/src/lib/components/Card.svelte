<script lang="ts">
  interface Props {
    padding?: boolean;
    hoverable?: boolean;
    children?: any;
  }

  let {
    padding = true,
    hoverable = false,
    children
  }: Props = $props();
</script>

<div 
  class="card"
  class:p-6={padding}
  class:card-hoverable={hoverable}
>
  {@render children?.()}
</div>
<script lang="ts">
  interface Props {
    padding?: boolean;
    hoverable?: boolean;
    class?: string;
    children?: any;
  }

  let {
    padding = true,
    hoverable = false,
    class: className = '',
    children
  }: Props = $props();
</script>

<div 
  class="card {className}"
  class:p-6={padding}
  class:card-hoverable={hoverable}
>
  {@render children?.()}
</div>
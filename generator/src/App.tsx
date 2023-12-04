import type { Component } from "solid-js";

import Canvas from "@/components/Canvas";
import Header from "@/components/Header";

const App: Component = () => {
  return (
    <div class="flex flex-col h-[100dvh] bg-light">
      <Header />
      <Canvas />
    </div>
  );
};

export default App;

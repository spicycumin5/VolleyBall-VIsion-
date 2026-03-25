  // toggle component
  
  const Toggle = ({ on, onToggle, label, color }) => (
    <button
      onClick={onToggle}
      className="flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all duration-150"
      style={{
        background: on ? color + "22" : "transparent",
        border: `1.5px solid ${on ? color : "#555"}`,
        color: on ? color : "#888",
        opacity: on ? 1 : 0.6,
      }}
    >
      <span
        className="w-2 h-2 rounded-full transition-all duration-150"
        style={{ background: on ? color : "#555" }}
      />
      {label}
    </button>
  );

export default Toggle;
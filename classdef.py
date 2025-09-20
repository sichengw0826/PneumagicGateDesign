import bpy

class CuttedFeatures:
    # Class‐level flag indicating whether a removal operation has been committed
    isCommitted: bool = False

    @classmethod
    def commit_removal(cls, cut_object: bpy.types.Object, cutting_object: bpy.types.Object) -> None:
        """
        Perform any removal logic here (e.g., Boolean difference),
        then mark the operation as committed.
        """
        # Example: apply Boolean modifier (difference) on cut_object using cutting_object
        mod = cut_object.modifiers.new(name="BooleanCut", type='BOOLEAN')
        mod.operation = 'DIFFERENCE'
        mod.object = cutting_object
        bpy.context.view_layer.objects.active = cut_object
        bpy.ops.object.modifier_apply(modifier=mod.name)

        cls.isCommitted = True
import bpy
import mathutils
import bmesh
import math
from typing import Iterable, Sequence, Optional


def create_cube(x_len, y_len, z_len, location=(0, 0, 0)):
    """
    Create a cube with the given dimensions.
    Note: Blender's default cube is 2x2x2 so we adjust its scale accordingly.
    """
    bpy.ops.mesh.primitive_cube_add(enter_editmode=False, location=location)
    cube = bpy.context.active_object
    cube.scale = (x_len / 2, y_len / 2, z_len / 2)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return cube

def get_world_edge_dict(obj, precision=6):
    """
    Given a mesh object, returns a dictionary where:
      - The keys are sorted tuples of the two endpoint coordinates in world space,
        with each coordinate rounded to the specified precision.
      - The values are the corresponding mesh edge objects.
    
    This canonical representation ensures that edges can be compared regardless of
    the order of their vertices.
    """
    world_matrix = obj.matrix_world
    mesh = obj.data
    
    # Get world-space coordinates for each vertex.
    vertices = [world_matrix @ v.co for v in mesh.vertices]
    
    edge_dict = {}
    for edge in mesh.edges:
        # Get the two vertices of the edge.
        v1 = vertices[edge.vertices[0]]
        v2 = vertices[edge.vertices[1]]
        # Round coordinates for precision.
        coord1 = (round(v1.x, precision), round(v1.y, precision), round(v1.z, precision))
        coord2 = (round(v2.x, precision), round(v2.y, precision), round(v2.z, precision))
        # Sort the two endpoints to get a canonical representation.
        sorted_coords = tuple(sorted([coord1, coord2]))
        edge_dict[sorted_coords] = edge
    return edge_dict

def compare_boolean_union_edges(objA, objB, union_obj, precision=6):
    """
    Given two original mesh objects (objA and objB) and a third object (union_obj)
    that is the result of a boolean union of objA and objB, this function compares
    their edges and returns a list of edge objects from the union object that appear
    only in the union (i.e. edges that are not present in either objA or objB).
    
    Edges are compared based on their world-space vertex coordinates (with rounding).
    """
    # Build dictionaries for the original objects.
    edges_A = get_world_edge_dict(objA, precision)
    edges_B = get_world_edge_dict(objB, precision)
    original_edge_keys = set(edges_A.keys()).union(edges_B.keys())
    
    # Build dictionary for the union object.
    union_edges = get_world_edge_dict(union_obj, precision)
    
    # Identify the keys (canonical edges) that exist only in the union object.
    unique_edge_keys = set(union_edges.keys()).difference(original_edge_keys)
    
    # Return the corresponding edge objects from the union object.
    return unique_edge_keys #[union_edges[key] for key in unique_edge_keys]



def move_edge_by_coords(translation_val, edge_coords, tolerance=1e-6):
    """
    Moves an edge in the active mesh object by a given translation.
    
    Parameters:
      translation : mathutils.Vector
          The translation vector by which to move the edge.
      edge_coords : list of tuple
          A list containing two tuples, each representing the (x, y, z) coordinates
          of the endpoints of the edge (in local object space). Example:
          [(x1, y1, z1), (x2, y2, z2)]
      tolerance : float
          The tolerance for matching vertex coordinates.
    """
    # Get the active object (assumed to be a mesh)
    translation = mathutils.Vector(translation_val)
    
    obj = bpy.context.active_object
    
    # Switch to Edit Mode
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Create a bmesh from the active object's mesh data
    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    
    # Prepare the provided coordinates as mathutils.Vector and sort them for canonical ordering
    provided_coords = sorted([mathutils.Vector(coord) for coord in edge_coords],
                             key=lambda v: (v.x, v.y, v.z))
    
    edge_found = False
    for edge in bm.edges:
        # Get the two vertex coordinates of the edge (local space)
        v1 =  obj.matrix_world @ edge.verts[0].co
        v2 =  obj.matrix_world @ edge.verts[1].co
        # Sort the vertex coordinates to match the provided order
        edge_coords_sorted = sorted([v1, v2], key=lambda v: (v.x, v.y, v.z))
        #print(edge_coords_sorted)
        
        # Check if both endpoints match the provided coordinates within the tolerance
        if (provided_coords[0] - edge_coords_sorted[0]).length < tolerance and \
           (provided_coords[1] - edge_coords_sorted[1]).length < tolerance:
            # Move both vertices of the matched edge by the translation vector
            for vert in edge.verts:
                vert.co += translation
            edge_found = True
            break

    if not edge_found:
        print("Edge not found matching the given coordinates.")
    else:
        print("Edge moved.")
    # Update the mesh and switch back to Object Mode
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

def create_hinge(v_start,v_end,hinge_width,hinge_thickness,hinge_gap_w,hinge_chamfer_ang,
                    thickness,hinge_empty_ratio=0.8):
    #hinge_empty_ratio = 0.8
    hinge_connection_min_w = 2
    #v_start = mathutils.Vector(start_pos)
    #v_end = mathutils.Vector(end_pos)
    v_full_hinge = v_end-v_start
    len_hinge = v_full_hinge.length
    
    # Create the gap between sheets and the wider hinge part
    gap_obj = create_cube(len_hinge, hinge_gap_w, thickness-hinge_thickness*0.5, location=(0, 0, (thickness-hinge_thickness*0.5)/2))
    gap_obj_copy = create_cube(len_hinge, hinge_gap_w, thickness, location=(0, 0, thickness/2))
    hinge_obj = create_cube(len_hinge, hinge_width, hinge_thickness, location=(0, 0, thickness-hinge_thickness/2))
    bool_mod = gap_obj.modifiers.new(name="create_profile", type='BOOLEAN')
    bool_mod.operation = 'UNION'
    bool_mod.object = hinge_obj
    bpy.context.view_layer.objects.active = gap_obj
    bpy.ops.object.modifier_apply(modifier=bool_mod.name)
    
    # Find two longest newly created edges and apply chamfer
    chamfer_dz = math.tan(math.radians(hinge_chamfer_ang))*(hinge_width-hinge_gap_w)/2
    new_edges = list(compare_boolean_union_edges(gap_obj_copy, hinge_obj, gap_obj, precision=6))
    edge_len_list = []
    for edge in new_edges:
        edge_vct = mathutils.Vector(edge[1])-mathutils.Vector(edge[0])
        edge_len_list.append(edge_vct.length)
    edge_ind = sorted(range(len(edge_len_list)), key=lambda i: edge_len_list[i])
    edge_ind.reverse()
    for i in range(2):
        print(new_edges[edge_ind[i]])
        move_edge_by_coords((0, 0, -chamfer_dz), new_edges[edge_ind[i]], tolerance=1e-5)   
        
    # If necessary, create hole on the hinge to reduce stiffness
    if len_hinge*(1-hinge_empty_ratio)>= hinge_connection_min_w:
        bool_mod = gap_obj.modifiers.new(name="create_profile", type='BOOLEAN')
        bool_mod.operation = 'UNION'
        hole_len = hinge_empty_ratio*len_hinge
        hole_obj = create_cube(hole_len, hinge_width, hinge_thickness*1.2, location=(0, 0, thickness+hinge_thickness/2))
        bool_mod.object = hole_obj
        bpy.context.view_layer.objects.active = gap_obj
        bpy.ops.object.modifier_apply(modifier=bool_mod.name)
        bpy.data.objects.remove(hole_obj, do_unlink=True)
    
    # Translate and rotate to the 
    gap_obj.location = (v_start+v_end)/2-mathutils.Vector((0,0,hinge_thickness))
    rot_ang = math.atan2(v_full_hinge[1],v_full_hinge[0])
    gap_obj.rotation_euler = (0.0, 0.0, rot_ang)
    
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)
    
    
    bpy.data.objects.remove(hinge_obj, do_unlink=True)
    bpy.data.objects.remove(gap_obj_copy, do_unlink=True)
    
    return gap_obj

def create_through_rec_hole(ctr,x_len,y_len,rot,target):
    z_list = []
    for vertex in target.bound_box:
        z_list.append(vertex[2])
    hole_dz = max(z_list)-min(z_list)
    hole_dz = hole_dz*1.01
    hole_box = create_cube(x_len, y_len, hole_dz, location=(ctr[0], ctr[1], (max(z_list)+min(z_list))/2))
    return hole_box

class BodyFeatures:
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
        bpy.data.objects.remove(cutting_object, do_unlink=True)

        cls.isCommitted = True
    
    def commit_merger(cls, merge_object: bpy.types.Object, merging_object: bpy.types.Object) -> None:
        """
        Perform any merger logic here (e.g., Boolean union),
        then mark the operation as committed.
        """
        # Example: apply Boolean modifier (difference) on cut_object using cutting_object
        mod = merge_object.modifiers.new(name="BooleanUni", type='BOOLEAN')
        mod.operation = 'UNION'
        mod.solver = 'FAST'
        mod.object = merging_object
        bpy.context.view_layer.objects.active = merge_object
        bpy.ops.object.modifier_apply(modifier=mod.name)
        bpy.data.objects.remove(merging_object, do_unlink=True)

        cls.isCommitted = True
"""
def commit_removal(cut_object, cutting_object):
    bool_mod = cut_object.modifiers.new(name="make_hole",type='BOOLEAN')
    bool_mod.operation = 'DIFFERENCE'
    bool_mod.object = cutting_object
    bpy.context.view_layer.objects.active = cut_object
    bpy.ops.object.modifier_apply(modifier=bool_mod.name)
    bpy.data.objects.remove(cutting_object, do_unlink=True)
"""

class Hinge(BodyFeatures):
    def __init__(self,startPt,endPt,segments,targetObject):
        """
        startPt, endPt: two 3D coordinates defining hinge endpoints
        segments: list of coordinate‑pair tuples defining hinge geometry
        targetObject: the Blender object to which this hinge applies
        """
        self.startPt = startPt
        self.endPt = endPt
        self.segments = segments
        self.targetObject = targetObject

class PathFeatures(BodyFeatures):
    """
    Utility base class for creating path-driven features that use a trapezoidal profile.
    """

    def __init__(
        self,
        base_object: bpy.types.Object,
        center: Sequence[float],
        path_points: Iterable[Sequence[float]],
        *,
        ctr_z: float = 0,
        segment_length: float = 1.0,
        base_width: float = 2.0,
        top_width: Optional[float] = None,
        height: float = 1.0,
        array_offset: float = 1.0,
        merge_vertices: bool = False,
        name_prefix: str = "PathFeature",
    ) -> None:
        self.base_object = base_object
        self.ctr_z = float(ctr_z)
        self.center_xy = mathutils.Vector((center[0], center[1]))
        self.center = mathutils.Vector((center[0], center[1], self.ctr_z))
        self.segment_length = segment_length
        self.base_width = base_width
        self.top_width = top_width if top_width is not None else base_width
        self.height = height
        self.array_offset = array_offset
        self.merge_vertices = merge_vertices
        self.name_prefix = name_prefix
        self.iscomitted: bool = False

        self._validate_dimensions()
        self.path_points = self._prepare_path_points(path_points)
        self.profile_object = self._create_trapezoid_profile()
        self.path_curve = self._create_path_curve()
        self._apply_path_modifiers()

    def _validate_dimensions(self) -> None:
        if self.segment_length <= 0 or self.base_width <= 0 or self.height <= 0:
            raise ValueError("segment_length, base_width, and height must be positive.")
        if self.top_width <= 0:
            raise ValueError("top_width must be positive.")

    def _prepare_path_points(
        self, points: Iterable[Sequence[float]]
    ) -> list[mathutils.Vector]:
        prepared = [mathutils.Vector((pt[0], pt[1], self.ctr_z)) for pt in points]
        if not prepared:
            raise ValueError("path_points must contain at least one point.")

        if (prepared[0] - self.center).length > 1e-6:
            prepared.insert(0, self.center.copy())

        return prepared

    def _create_trapezoid_profile(self) -> bpy.types.Object:
        profile = create_cube(
            self.segment_length,
            self.base_width,
            self.height,
            location=self.center,
        )
        profile.name = f"{self.name_prefix}_Profile"
        
        print(self.center.z)

        half_len = self.segment_length / 2
        half_width = self.base_width / 2
        half_height = self.height / 2
        top_z = self.center.z + half_height
        delta = (self.base_width - self.top_width) / 2
        
        if abs(delta) > 1e-7:
            bpy.context.view_layer.objects.active = profile
            profile.select_set(True)

            positive_edge = [
                (self.center.x - half_len, self.center.y + half_width, top_z),
                (self.center.x + half_len, self.center.y + half_width, top_z),
            ]
            negative_edge = [
                (self.center.x - half_len, self.center.y - half_width, top_z),
                (self.center.x + half_len, self.center.y - half_width, top_z),
            ]
            move_edge_by_coords((0, -delta, 0), positive_edge, tolerance=1e-5)
            move_edge_by_coords((0, delta, 0), negative_edge, tolerance=1e-5)

        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        
        return profile

    def _create_path_curve(self) -> bpy.types.Object:
        curve_data = bpy.data.curves.new(f"{self.name_prefix}_Path", type="CURVE")
        curve_data.dimensions = "3D"
        spline = curve_data.splines.new(type="POLY")
        spline.points.add(len(self.path_points) - 1)

        for idx, point in enumerate(self.path_points):
            spline.points[idx].co = (point.x, point.y, point.z, 1.0)

        curve_obj = bpy.data.objects.new(f"{self.name_prefix}_PathObj", curve_data)
        bpy.context.collection.objects.link(curve_obj)
        return curve_obj

    def _apply_path_modifiers(self) -> None:
        array_mod = self.profile_object.modifiers.new(
            name=f"{self.name_prefix}_Array", type="ARRAY"
        )
        array_mod.fit_type = "FIT_CURVE"
        array_mod.curve = self.path_curve
        array_mod.use_relative_offset = True
        array_mod.relative_offset_displace = (self.array_offset, 0.0, 0.0)
        array_mod.use_merge_vertices = self.merge_vertices

        curve_mod = self.profile_object.modifiers.new(
            name=f"{self.name_prefix}_Curve", type="CURVE"
        )
        curve_mod.object = self.path_curve
        curve_mod.deform_axis = "POS_X"
        
        bpy.ops.object.modifier_apply(modifier=array_mod.name)
        bpy.ops.object.modifier_apply(modifier=curve_mod.name)

    def _remove_path_curve(self) -> None:
        if self.path_curve and self.path_curve.name in bpy.data.objects:
            bpy.data.objects.remove(self.path_curve, do_unlink=True)
        self.path_curve = None
    '''
    def commit_merger(self) -> None:
        mod = self.base_object.modifiers.new(
            name=f"{self.name_prefix}_Union", type="BOOLEAN"
        )
        mod.operation = "UNION"
        mod.object = self.profile_object
        bpy.context.view_layer.objects.active = self.base_object
        bpy.ops.object.modifier_apply(modifier=mod.name)

        if self.profile_object and self.profile_object.name in bpy.data.objects:
            bpy.data.objects.remove(self.profile_object, do_unlink=True)
        self.profile_object = None

        self._remove_path_curve()
        self.iscomitted = True
    '''


class TunnelCut(PathFeatures):
    """
    Create an isosceles trapezoid tunnel along a path and subtract it from the base object.
    """

    def __init__(
        self,
        base_object: bpy.types.Object,
        center: Sequence[float],
        path_points: Iterable[Sequence[float]],
        *,
        segment_length: float = 1.0,
        base_width: float = 2.0,
        top_width: float = 1.0,
        height: float = 1.0,
        array_offset: float = 1.0,
        merge_vertices: bool = False,
        name_prefix: str = "Tunnel",
    ) -> None:
        super().__init__(
            base_object,
            center,
            path_points,
            ctr_z=center[2],
            segment_length=segment_length,
            base_width=base_width,
            top_width=top_width,
            height=height,
            array_offset=array_offset,
            merge_vertices=merge_vertices,
            name_prefix=name_prefix,
        )

        self.commit_removal(self.base_object, self.profile_object)
        self.profile_object = None
        self._remove_path_curve()
        self.iscomitted = True


class SurfacePath(PathFeatures):
    """
    Create a rectangular relief path on the top or bottom surface of a base object.
    """

    def __init__(
        self,
        base_object: bpy.types.Object,
        width: float,
        relief_height: float,
        surface: str,
        path_points: Iterable[Sequence[float]],
        *,
        segment_length: float = 1.0,
        array_offset: float = 1.0,
        merge_vertices: bool = False,
        name_prefix: str = "SurfacePath",
    ) -> None:
        ref_points = [mathutils.Vector(corner) for corner in base_object.bound_box]
        world_points = [base_object.matrix_world @ ref for ref in ref_points]
        z_values = [pt.z for pt in world_points]
        surface_z = max(z_values) if surface == "top" else min(z_values)
        
        path_point_list = [(pt[0], pt[1],0) for pt in path_points]

        if relief_height <= 0 or width <= 0:
            raise ValueError("width and relief_height must be positive.")

        surface = surface.lower()
        if surface not in {"top", "bottom"}:
            raise ValueError('surface must be either "top" or "bottom".')

        if not path_point_list:
            raise ValueError("path_points must contain at least one point.")

        center_xy = path_point_list[0]
        height = 2.0 * relief_height
        
        #print(surface_z)

        super().__init__(
            base_object,
            center_xy,
            path_point_list,
            ctr_z=0,
            segment_length=segment_length,
            base_width=width,
            top_width=width,
            height=height,
            array_offset=array_offset,
            merge_vertices=merge_vertices,
            name_prefix=name_prefix,
        )
        
        self.profile_object.location.z += surface_z #Ensure alignment: make paths at z=0, then translate to desired z
        
        self.commit_merger(self.base_object, self.profile_object)
        self.profile_object = None
        self._remove_path_curve()
        self.iscomitted = True
    

thickness = 1.8

hinge_width = 0.6
hinge_thickness = 0.12
hinge_gap_w = 0.1
hinge_chamfer_ang = 45
hinge_empty_ratio = 0.9
hinge_connection_min_w = 2

base_len = 54
base_wdt = 34

fold_len_ratio = 11/54
fin_width_ratio = 12/34

# Useful values
fold_x = -(0.5-fold_len_ratio)*base_len
fin_start = mathutils.Vector((fold_x,-base_wdt/2*1.01,0))
find_end = mathutils.Vector((fold_x,base_wdt*(-0.5+fin_width_ratio),0))
opening_ctr = mathutils.Vector((fold_x,0.25*base_wdt,0))
# Operations
base_sheet = create_cube(base_len, base_wdt, thickness, location=(0, 0, 0))

cutting_list = []

cutting_list.append(create_hinge(fin_start,find_end,hinge_width,hinge_thickness,hinge_gap_w,hinge_chamfer_ang,thickness))
cutting_list.append(create_through_rec_hole(opening_ctr,5,14,0,base_sheet))
SurfacePath(base_sheet,0.8,0.2,'top',[(0,0,0),(0,5,0),(0,8,0),(5,8,0),(7,8,0)])
cutting_list.append(TunnelCut(base_sheet,[0,0,0],[(0,0,0),(0,5,0),(0,8,0),(5,8,0),(7,8,0)]))

'''
def build_default_sheet():
    """Generate the default sheet geometry with hinges and rectangular cutouts."""
    fold_x = -(0.5-fold_len_ratio)*base_len
    fin_start = mathutils.Vector((fold_x,-base_wdt/2*1.01,0))
    find_end = mathutils.Vector((fold_x,base_wdt*(-0.5+fin_width_ratio),0))
    opening_ctr = mathutils.Vector((fold_x,0.25*base_wdt,0))

    base_sheet = create_cube(base_len, base_wdt, thickness, location=(0, 0, 0))
    cutting_list = [
        create_hinge(fin_start,find_end,hinge_width,hinge_thickness,hinge_gap_w,hinge_chamfer_ang,thickness),
        create_through_rec_hole(opening_ctr,5,14,0,base_sheet),
    ]

    for obj in cutting_list:
        commit_removal(base_sheet, obj)

    return base_sheet

if __name__ == "__main__":
    build_default_sheet()
'''